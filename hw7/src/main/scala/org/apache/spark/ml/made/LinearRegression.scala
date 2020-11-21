package org.apache.spark.ml.made

import java.util

import breeze.linalg._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasMaxIter, HasPredictionCol, HasStepSize}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, MetadataUtils, SchemaUtils}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.mllib
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.functions._

trait LinearRegressionParams extends HasLabelCol with HasFeaturesCol with HasPredictionCol
  with HasMaxIter with HasStepSize {

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
    } else {
      SchemaUtils.appendColumn(schema, StructField(getPredictionCol, new VectorUDT()))
    }
    if (schema.fieldNames.contains($(labelCol))) {
      SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)
    } else {
      SchemaUtils.appendColumn(schema, StructField(getLabelCol, DoubleType))
    }
    schema
  }
}


class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  setDefault(maxIter, 100)

  def setStepSize(value: Double): this.type = set(stepSize, value)

  setDefault(stepSize, 0.1)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val assembler = new VectorAssembler()
      .setInputCols(Array("intercept", $(featuresCol), $(labelCol)))
      .setOutputCol("feats")
    val features = assembler
      .transform(dataset.withColumn("intercept", lit(1)))
      .select("feats").as[Vector]

    val n_features = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var coefficients = breeze.linalg.DenseVector.rand[Double](n_features + 1)
    for (_ <- 0 to $(maxIter)) {
      val summary = features.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val X = v.asBreeze(0 until coefficients.size).toDenseVector
          val y = v.asBreeze(-1)
          val grads = X * (breeze.linalg.sum(X * coefficients) - y)
          summarizer.add(mllib.linalg.Vectors.fromBreeze(grads))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      coefficients = coefficients - $(stepSize) * summary.mean.asBreeze
    }

    copyValues(new LinearRegressionModel(
      Vectors.fromBreeze(coefficients(1 until coefficients.size)).toDense,
      coefficients(0))
    ).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val coefficients: DenseVector,
                                           val intercept: Double) extends Model[LinearRegressionModel]
  with LinearRegressionParams with MLWritable {

  private[made] def this(coefficients: DenseVector, intercept: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), coefficients, intercept)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(uid, coefficients, intercept))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_predict",
      (x: Vector) => {
        Vectors.fromBreeze(breeze.linalg.DenseVector(coefficients.asBreeze.dot(x.asBreeze) + intercept))
      })

    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      sqlContext.createDataFrame(Seq(coefficients -> intercept)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val coefficients = vectors.select(vectors("_1").as[Vector]).first()
      val intercept = vectors.select(vectors("_2")).first().getDouble(0)

      val model = new LinearRegressionModel(coefficients.toDense, intercept)
      metadata.getAndSetParams(model)
      model
    }
  }
}

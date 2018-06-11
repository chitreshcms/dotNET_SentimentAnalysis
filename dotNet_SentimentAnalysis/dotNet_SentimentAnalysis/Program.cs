using System;
//Following have to be implemented for the Microsoft ML framework.
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Threading.Tasks;
namespace dotNet_SentimentAnalysis
{
    class Program
    {
       /* Global variables to hold the path to the recently downloaded files:
        1)  _dataPath has the path to the dataset used to train the model.
        2)  _testDataPath has the path to the dataset used to evaluate the model.
        3)  _modelPath has the path where the trained model is saved. */
        const string _dataPath = @".\Data\wikipedia-detox-250-line-data.tsv";
        const string _testDataPath = @".\Data\wikipedia-detox-250-line-test.tsv";
        const string _modelPath = @".\Data\Model.zip";

        static async Task Main(string[] args)
        {
            var model = await Train();
            Evaluate(model);
            Predict(model);

        }

        public static async Task<PredictionModel<SentimentData, SentimentPrediction>> Train()
        {
            //1- Ingest the data
            /* Initialize a new instance of LearningPipeline that will 
             * include the data loading, 
             * data processing/featurization, and model. */
            var pipeline = new LearningPipeline();

            /* The TextLoader<TInput> object is the first part of the pipeline, 
                and loads the training file data. */
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());

            // 2- Data preprocess and feature engineering
            /*Apply a TextFeaturizer to convert the SentimentText column 
             * into a numeric vector called 
             * Features used by the machine learning algorithm. 
            This is the preprocessing / featurization step.
            Using additional components available in ML.NET can enable better 
            results with your model. */

            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));

            //3 -Choose a learning algorithm

            /* The FastTreeBinaryClassifier object is a decision tree learner 
             * you'll use in this pipeline. Similar to the featurization step,
             * trying out different learners available in ML.NET and changing their 
             * parameters leads to different results. For tuning, you can set 
             * hyperparameters like NumTrees, NumLeaves, and MinDocumentsInLeafs. 
             * These hyperparameters are set before anything affects the model and are model-specific.
             * They're used to tune the decision tree for performance, 
             * so larger values can negatively impact performance. */

            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            //4-Train the model
            /*You train the model, PredictionModel< TInput,TOutput >,
            based on the dataset that has been loaded and transformed. pipeline.
            Train<SentimentData, SentimentPrediction>() trains the pipeline(loads the data,
            trains the featurizer and learner).The experiment is not executed until this happens. */

            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();

            //5-Save and Return the model trained to use for evaluation
            /*At this point, you have a model that can be integrated into any of your existing or new .NET applications. 
             * To save your model to a .zip file before returning */
            await model.WriteAsync(_modelPath);

            return model;

        }

        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();

            var evaluator = new BinaryClassificationEvaluator();

            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }

        public static void Predict(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            IEnumerable<SentimentData> sentiments = new[]
{
        new SentimentData
        {
            SentimentText = "Please refrain from adding nonsense to Wikipedia."
        },
        new SentimentData
         {
        SentimentText = "He is the best, and the article should say that."
        }
        };
            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);

            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");

            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

            foreach (var item in sentimentsAndPredictions)
            {
                //bool flag = false;
                //if (item.prediction.Sentiment > 0)
                //    flag = true;
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
            }
            Console.WriteLine();
            Console.ReadKey();
        }
    }
}

﻿using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Api;

namespace dotNet_SentimentAnalysis
{
    public class SentimentData
    {
        [Column(ordinal:"0",name:"Label")]
        public float Sentiment;

        [Column(ordinal:"1")]
        public string SentimentText;
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Sentiment;
    }
}

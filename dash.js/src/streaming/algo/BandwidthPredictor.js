MediaPlayer.dependencies.BandwidthPredictor = function () {
    "use strict";

    var pastThroughput = [],
    pastDownloadTime = [],
    // Xiaoqi_final
    bandwidthEstLog = [],
    // Xiaoqi_final
    // Xiaoqi_cr
    horizon = 5, // number of chunks considered
    // Xiaoqi_cr

    predict = function (lastRequested, lastQuality, lastHTTPRequest) {
    var self = this,
    bandwidthEst = 0,
    lastDownloadTime,
    lastThroughput,
    lastChunkSize,
    // Xiaoqi_cr
    // horizon = 5, // number of chunks considered
    // Xiaoqi_cr
    tmpIndex,
    tmpSum = 0,
    tmpDownloadTime = 0;

    // self.debug.log("----------ENTERING predictBandwidth");

    // First, log last download time and throughput
    if (lastHTTPRequest && lastRequested >= 0) {
        // self.debug.log("----------ENTERING predictBandwidth, lastHTTPRequest not empty");
        lastDownloadTime = (lastHTTPRequest.tfinish.getTime() - lastHTTPRequest.tresponse.getTime()) / 1000; //seconds
        if (lastDownloadTime <0.1) {
        lastDownloadTime = 0.1;
        }
        lastChunkSize = self.vbr.getChunkSize(lastRequested, lastQuality);
        lastThroughput = lastChunkSize*8/lastDownloadTime/1000;
        // Log last chunk
        pastThroughput[lastRequested] = lastThroughput;
        pastDownloadTime[lastRequested] = lastDownloadTime;
        // debug
        self.debug.log("----------BWPredict lastChunk="+lastRequested+", downloadTime="+lastDownloadTime+"s, lastThroughput="+lastThroughput+"kb/s, lastChunkSize="+lastChunkSize);
    }

    // Next, predict future bandwidth
    if (pastThroughput.length === 0) {
        return 0;
    } else {
        tmpIndex = Math.max(0, lastRequested + 1 - horizon);
        tmpSum = 0;
        tmpDownloadTime = 0;
        for (var i = tmpIndex; i<= lastRequested; i++) {
        tmpSum = tmpSum + pastDownloadTime[i]/pastThroughput[i];
        tmpDownloadTime = tmpDownloadTime + pastDownloadTime[i];
        }
        bandwidthEst = tmpDownloadTime/tmpSum;
        // Xiaoqi_final
        bandwidthEstLog[lastRequested] = bandwidthEst;
        // Xiaoqi_final
        self.debug.log("----------BWPredict: bwEst="+ bandwidthEst+", tmpIndex="+tmpIndex+", tmpDownloadTime="+tmpDownloadTime);
        return bandwidthEst;
    }
    },

    // Xiaoqi_cr
    getPredictionError = function (lastRequested) {
    var self = this,
    tmpIndex,
    tmpError = 0,
    tmpSum = 0,
    maxError = 0;

    if (pastThroughput.length <= 1) { // not enough data
        return 0;
    } else {
        tmpIndex = Math.max(1, lastRequested + 1 - horizon);
        for (var i = tmpIndex; i<= lastRequested; i++) {
        // error
        // tmpError = (bandwidthEstLog[i-1] - pastThroughput[i])/pastThroughput[i]; // overprediction percentage
        // abs error
        tmpError = Math.abs((bandwidthEstLog[i-1] - pastThroughput[i])/pastThroughput[i]); // overprediction abs percentage
        // max error
        if (tmpError > maxError) {
            maxError = tmpError;
        }
        // // mean error
        // tmpSum = tmpSum + tmpError;
        }
        // meanError = tmpSum/(lastRequested-tmpIndex+1);
        self.debug.log("----------BWPredict: maxError="+maxError);
        return maxError;
    }
    },

    getMultiStepPredictionError = function (lastRequested, steps) {
    var self = this,
    tmpIndex,
    tmpError = 0,
    maxError = 0;

    if (pastThroughput.length <= steps) { // not enough data
        return 0;
    } else {
        tmpIndex = Math.max(1, lastRequested + 1 - horizon);
        for (var i = tmpIndex; i<= lastRequested; i++) {
        if (i-steps < 0) {
            tmpError = 0;
        } else {
            // error
            // tmpError = (bandwidthEstLog[i-1] - pastThroughput[i])/pastThroughput[i]; // overprediction percentage
            // abs error
            tmpError = Math.abs((bandwidthEstLog[i-steps] - pastThroughput[i])/pastThroughput[i]); // overprediction abs percentage
        }
        // max error
        if (tmpError > maxError) {
            maxError = tmpError;
        }
        }
        // self.debug.log("----------BWPredict, MultiStep: maxError="+maxError+", Step:"+steps);
        return maxError;
    }
    },

    getCombinedPredictionError = function (lastRequested) {
    var self = this,
    tmpError = 0,
    maxError = 0;
    for (var steps = 1; steps<= 5; steps++) {
        tmpError = getMultiStepPredictionError(lastRequested, steps);
        if (tmpError > maxError) {
        maxError = tmpError;
        }
    }
    return maxError;
    };
    // Xiaoqi_cr

    return {
        debug: undefined,
        //abrRulesCollection: undefined,
        //manifestExt: undefined,
        metricsModel: undefined,
    metricsEst: undefined,
    vbr: undefined,

    getBitrate: function () {
            return 0;
        },
    getPastThroughput: function() {
        return pastThroughput;
    },
    getBandwidthEstLog: function() {
        return bandwidthEstLog;
    },
    predictBandwidth: predict,
    getPredictionError: getPredictionError,
    getMultiStepPredictionError:  getMultiStepPredictionError,
    getCombinedPredictionError: getCombinedPredictionError
    // predictWorstCaseBandwidth: function (lastRequested, lastQuality, lastHTTPRequest) {
    //     var bandwidthEst,
    //     maxError,
    //     bandwidthEstWC;

    //     bandwidthEst = predict(lastRequested, lastQuality, lastHTTPRequest);
    //     maxError = getPredictionError(lastRequested);
    //     bandwidthEstWC = bandwidthEst/(1+maxError);
    //     return bandwidthEstWC;
    // }
    };
};

MediaPlayer.dependencies.BandwidthPredictor.prototype = {
    constructor: MediaPlayer.dependencies.BandwidthPredictor
};

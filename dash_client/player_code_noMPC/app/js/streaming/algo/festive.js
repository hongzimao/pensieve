MediaPlayer.dependencies.festive = function () {
    "use strict";

    var alpha = 12, 
    qualityLog = [],
    horizon = 5,
    switchUpCount = 0,
    switchUpThreshold = [0,1,2,3,4],
    p = 0.85,
    lastIndex = -1,
    // stabilityScore = 0,
    // efficiencyScore = 0,

    getStabilityScore = function (b, b_ref, b_cur) {
	var score = 0,
	n = 0;
	if (lastIndex >= 1) {
	    for (var i = Math.max(0, lastIndex + 1 - horizon); i<= lastIndex - 1; i++) {
		if (qualityLog[i] != qualityLog[i+1]) {
		    n = n + 1;
		}
	    }
	}
	if (b != b_cur) {
	    n = n + 1;
	}
	score = Math.pow(2,n);
	return score;
    },

    getEfficiencyScore = function (b, b_ref, w, bitrateArray) {
	var score = 0;
	score = Math.abs( bitrateArray[b]/Math.min(w, bitrateArray[b_ref]) - 1 );
	return score;
    },

    getCombinedScore = function (b, b_ref, b_cur, w, bitrateArray) {
	var stabilityScore = 0,
	efficiencyScore = 0,
	totalScore = 0;
	// compute
	stabilityScore = getStabilityScore(b, b_ref, b_cur);
	efficiencyScore = getEfficiencyScore(b, b_ref, w, bitrateArray);
	totalScore = stabilityScore + alpha*efficiencyScore;
	return totalScore;	
    },

    getBitrate = function (prevQuality, bufferLevel, bwPrediction, lastRequested, bitrateArray) {
	var self = this, 
	bitrate = 0,
	tmpBitrate = 0,
	b_target = 0,
	b_ref = 0,
	b_cur = prevQuality,
	score_cur = 0,
	score_ref = 0;
	// TODO: implement FESTIVE logic
	// 1. log previous quality
	qualityLog[lastRequested] = prevQuality;
	lastIndex = lastRequested;
	// 2. compute b_target
	tmpBitrate = p*bwPrediction;
	for (var i = 4; i>=0; i--) { // todo: use bitrateArray.length
	    if (bitrateArray[i] <= tmpBitrate) {
	    	b_target = i;
	    	break;
	    }
	    b_target = i;
	}
	self.debug.log("-----FESTIVE: lastRequested="+lastRequested+", bwPrediction="+bwPrediction+", b_target="+b_target+", switchUpCount="+ switchUpCount);
	// 3. compute b_ref
	if (b_target > b_cur) {
	    switchUpCount = switchUpCount + 1;
	    if (switchUpCount > switchUpThreshold[b_cur]) {
		b_ref = b_cur + 1;
	    } else {
		b_ref = b_cur;
	    }
	} else if (b_target < b_cur) {
	    b_ref = b_cur - 1;
	    switchUpCount = 0;
	} else {
	    b_ref = b_cur;
	    switchUpCount = 0; // this means need k consecutive "up" to actually switch up
	}
	// 4. delayed update
	if (b_ref != b_cur) { // need to switch
	    // compute scores
	    score_cur = getCombinedScore(b_cur, b_ref, b_cur, bwPrediction, bitrateArray);
	    score_ref = getCombinedScore(b_ref, b_ref, b_cur, bwPrediction, bitrateArray);
	    if (score_cur <= score_ref) {
		bitrate = b_cur;
	    } else {
		bitrate = b_ref;
		if (bitrate > b_cur) { // clear switchupcount
		    switchUpCount = 0;
		}
	    }
	} else {
	    bitrate = b_cur;
	}
	self.debug.log("-----FESTIVE: bitrate="+bitrate+", b_ref="+b_ref+", b_cur="+b_cur);
	// 5. return
	return bitrate;
    };

    return {
        debug: undefined,
        abrRulesCollection: undefined,
        manifestExt: undefined,
        metricsModel: undefined,

	getBitrate: getBitrate
    };
};

MediaPlayer.dependencies.festive.prototype = {
    constructor: MediaPlayer.dependencies.festive
};

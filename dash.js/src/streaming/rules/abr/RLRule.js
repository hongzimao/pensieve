/**
 * The copyright in this software is being made available under the BSD License,
 * included below. This software may be subject to other third party and contributor
 * rights, including patent rights, and no such rights are granted under this license.
 *
 * Copyright (c) 2013, Dash Industry Forum.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation and/or
 *  other materials provided with the distribution.
 *  * Neither the name of Dash Industry Forum nor the names of its
 *  contributors may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY
 *  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 *  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 *  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 *  NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
import SwitchRequest from '../SwitchRequest';
import MediaPlayerModel from '../../models/MediaPlayerModel';
import AbrController from '../../controllers/AbrController';
import FactoryMaker from '../../../core/FactoryMaker';
import Debug from '../../../core/Debug';
import {HTTPRequest} from '../../vo/metrics/HTTPRequest';


function RLRule(config) {

    let instance;
    let context = this.context;
    let log = Debug(context).getInstance().log;

    let metricsModel = config.metricsModel;
    let dashMetrics = config.dashMetrics;

    let finished_requests = 0;

    let lastSwitchTime,
        mediaPlayerModel;

    function setup() {
        lastSwitchTime = 0;
        mediaPlayerModel = MediaPlayerModel(context).getInstance();
    }

    function execute (rulesContext, callback) {
        // defaults to wanting to make a new request
        var switchRequest = SwitchRequest(context).create(1000000, SwitchRequest.WEAK, {name: RLRule.__dashjs_factory_name})

        // counts the number of completed and oustanding requests
        let curr_reqs = dashMetrics.getHttpRequests(metricsModel.getReadOnlyMetricsFor('video'));
        let completed = [];
        let outstanding = [];
        for ( let i = 0; i < curr_reqs.length; i++ ) {
            let request = curr_reqs[i];
            if (request.type === HTTPRequest.MEDIA_SEGMENT_TYPE && request._tfinish && request.tresponse && request.trace) {
                completed.push(request);
            }
            if (request.type === HTTPRequest.MEDIA_SEGMENT_TYPE && !request._tfinish && request.tresponse && request.trace) {
                outstanding.push(request);
            }
        }
        console.log("NUMBER OF COMPLETE: " + completed.length + " AT " + Date.now());

        // only send request to RL server if the previous chunk has finished downloading (i.e., number of downloaded requests has gone up by 1)
        // if not, then specify that we don't want to send anything to the RL server
        if ( completed.length <= finished_requests ) {
            switchRequest = SwitchRequest(context).create(SwitchRequest.NO_CHANGE, SwitchRequest.WEAK, {name: RLRule.__dashjs_factory_name});
        } else {
            finished_requests = completed.length;
        }
        callback(switchRequest);
    }

    function reset() {
        lastSwitchTime = 0;
    }

    instance = {
        execute: execute,
        reset: reset
    };

    setup();

    return instance;
}

RLRule.__dashjs_factory_name = 'RLRule';
export default FactoryMaker.getClassFactory(RLRule);

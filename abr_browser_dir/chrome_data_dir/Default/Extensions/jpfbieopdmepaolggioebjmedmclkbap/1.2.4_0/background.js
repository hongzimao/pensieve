(function() {
    var cachingEnabled = false;
    var alertThrown = false;
    var clearRunning = false;

    var clearCache = (function() {
        if (!clearRunning) {
            //if (chrome.experimental != undefined && chrome.experimental.clear != undefined) {
            if (typeof(chrome.browsingData) !== 'undefined') {
                clearRunning = true;
                var millisecondsPerWeek = 1000 * 60 * 60 * 24 * 7;
                var oneWeekAgo = (new Date()).getTime() - millisecondsPerWeek;
                
                //Chrome 19:
                chrome.browsingData.removeCache({
                      "since": oneWeekAgo
                    }, function() {
                    clearRunning = false;
                });
            } else if (!alertThrown) {
                alertThrown = true;
                alert("Your browser does not support cache cleaning :(");
            }
        }
    });
    
    var enableCaching = (function() {
        cachingEnabled = true;
        chrome.browserAction.setIcon({path:"icon-off.png"});
        chrome.browserAction.setTitle({title:"Cache Killer disabled"});
        chrome.webRequest.onBeforeRequest.removeListener(clearCache);
    });
    
    var disableCaching = (function() {
        cachingEnabled = false;
        chrome.browserAction.setIcon({path:"icon-on.png"});
        chrome.browserAction.setTitle({title:"Cache Killer enabled"});
        chrome.webRequest.onBeforeRequest.addListener(clearCache, {urls: ["<all_urls>"]});
    });

    var flipStatus = (function() {
        if (cachingEnabled) {
            disableCaching();
        } else {
            enableCaching();
        }
    });

    chrome.browserAction.onClicked.addListener(flipStatus);
    
    if (localStorage && localStorage["turnOnByDefault"] && localStorage["turnOnByDefault"] === "on") {
        disableCaching();
    } else {
        enableCaching();
    }
    if (_.loaded()) {
        console.log('ready');
    }
})();
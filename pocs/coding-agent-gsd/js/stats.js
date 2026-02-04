var stats = {
    score: 0,
    lines: 0,
    level: 1,
    time: 0,
    piecesPlaced: 0,
    sessionStartTime: 0
};

function startSession() {
    stats.score = 0;
    stats.lines = 0;
    stats.level = 1;
    stats.time = 0;
    stats.piecesPlaced = 0;
    stats.sessionStartTime = performance.now();
}

function updatePiecePlaced() {
    stats.piecesPlaced++;
}

function getSessionTime() {
    return performance.now() - stats.sessionStartTime;
}

function formatTime(ms) {
    var totalSeconds = Math.floor(ms / 1000);
    var minutes = Math.floor(totalSeconds / 60);
    var seconds = totalSeconds % 60;
    var minStr = minutes < 10 ? '0' + minutes : '' + minutes;
    var secStr = seconds < 10 ? '0' + seconds : '' + seconds;
    return minStr + ':' + secStr;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        stats: stats,
        startSession: startSession,
        updatePiecePlaced: updatePiecePlaced,
        getSessionTime: getSessionTime,
        formatTime: formatTime
    };
}

var stats = {
    score: 0,
    lines: 0,
    level: 1,
    time: 0,
    piecesPlaced: 0,
    sessionStartTime: 0,
    actionsCount: 0,
    maxCombo: 0,
    b2bCount: 0,
    tSpinCount: 0
};

function startSession() {
    stats.score = 0;
    stats.lines = 0;
    stats.level = 1;
    stats.time = 0;
    stats.piecesPlaced = 0;
    stats.sessionStartTime = performance.now();
    stats.actionsCount = 0;
    stats.maxCombo = 0;
    stats.b2bCount = 0;
    stats.tSpinCount = 0;
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

function trackAction() {
    stats.actionsCount++;
}

function calculatePPS() {
    var timeMs = getSessionTime();
    if (timeMs === 0) return 0;
    return stats.piecesPlaced / (timeMs / 1000);
}

function calculateAPM() {
    var timeMs = getSessionTime();
    if (timeMs === 0) return 0;
    return stats.actionsCount / (timeMs / 60000);
}

function calculateEfficiency() {
    if (stats.actionsCount === 0) return 0;
    return (stats.piecesPlaced * 60) / stats.actionsCount * 100;
}

function getTetrisRate() {
    return 0;
}

function formatDecimal(num, decimals) {
    return num.toFixed(decimals);
}

function formatPercent(num) {
    return num.toFixed(1) + '%';
}

function updateComboStats(currentCombo) {
    if (currentCombo > stats.maxCombo) {
        stats.maxCombo = currentCombo;
    }
}

function incrementB2bCount() {
    stats.b2bCount++;
}

function updateTSpinStats(tSpinType) {
    if (tSpinType !== null) {
        stats.tSpinCount++;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        stats: stats,
        startSession: startSession,
        updatePiecePlaced: updatePiecePlaced,
        getSessionTime: getSessionTime,
        formatTime: formatTime,
        trackAction: trackAction,
        calculatePPS: calculatePPS,
        calculateAPM: calculateAPM,
        calculateEfficiency: calculateEfficiency,
        getTetrisRate: getTetrisRate,
        formatDecimal: formatDecimal,
        formatPercent: formatPercent,
        updateComboStats: updateComboStats,
        incrementB2bCount: incrementB2bCount,
        updateTSpinStats: updateTSpinStats
    };
}

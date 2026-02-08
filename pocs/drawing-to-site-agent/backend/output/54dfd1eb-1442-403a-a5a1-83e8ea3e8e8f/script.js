var timerEl = document.getElementById("timer");
var scoreBlueEl = document.getElementById("score-blue");
var scoreRedEl = document.getElementById("score-red");
var ballEl = document.getElementById("ball");
var btnStart = document.getElementById("btn-start");
var btnPause = document.getElementById("btn-pause");
var btnReset = document.getElementById("btn-reset");
var btnGoalBlue = document.getElementById("btn-goal-blue");
var btnGoalRed = document.getElementById("btn-goal-red");
var redCard = document.getElementById("red-card");
var blueCard = document.getElementById("blue-card");

var seconds = 54;
var interval = null;
var running = false;
var scoreBlue = 0;
var scoreRed = 2;
var dragging = false;
var dragOffsetX = 0;
var dragOffsetY = 0;

function formatTime(s) {
  var m = Math.floor(s / 60);
  var sec = s % 60;
  return m + ":" + (sec < 10 ? "0" + sec : sec);
}

function updateTimer() {
  timerEl.textContent = formatTime(seconds);
}

function updateScores() {
  scoreBlueEl.textContent = scoreBlue;
  scoreRedEl.textContent = scoreRed;
}

function flashScore(el) {
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
  setTimeout(function () {
    el.classList.remove("flash");
  }, 1000);
}

btnStart.addEventListener("click", function () {
  if (running) return;
  running = true;
  interval = setInterval(function () {
    seconds++;
    updateTimer();
  }, 1000);
});

btnPause.addEventListener("click", function () {
  running = false;
  clearInterval(interval);
});

btnReset.addEventListener("click", function () {
  running = false;
  clearInterval(interval);
  seconds = 0;
  scoreBlue = 0;
  scoreRed = 0;
  updateTimer();
  updateScores();
  ballEl.style.top = "30%";
  ballEl.style.left = "55%";
});

btnGoalBlue.addEventListener("click", function () {
  scoreBlue++;
  updateScores();
  flashScore(scoreBlueEl);
});

btnGoalRed.addEventListener("click", function () {
  scoreRed++;
  updateScores();
  flashScore(scoreRedEl);
});

redCard.addEventListener("click", function () {
  redCard.classList.toggle("active");
});

blueCard.addEventListener("click", function () {
  blueCard.classList.toggle("active");
});

ballEl.addEventListener("mousedown", function (e) {
  dragging = true;
  ballEl.classList.add("dragging");
  var rect = ballEl.getBoundingClientRect();
  dragOffsetX = e.clientX - rect.left;
  dragOffsetY = e.clientY - rect.top;
  e.preventDefault();
});

document.addEventListener("mousemove", function (e) {
  if (!dragging) return;
  var field = document.querySelector(".field");
  var fieldRect = field.getBoundingClientRect();
  var x = e.clientX - fieldRect.left - dragOffsetX;
  var y = e.clientY - fieldRect.top - dragOffsetY;
  x = Math.max(0, Math.min(x, fieldRect.width - 48));
  y = Math.max(0, Math.min(y, fieldRect.height - 48));
  ballEl.style.left = x + "px";
  ballEl.style.top = y + "px";
  ballEl.style.transform = "none";
});

document.addEventListener("mouseup", function () {
  dragging = false;
  ballEl.classList.remove("dragging");
});

ballEl.addEventListener("touchstart", function (e) {
  dragging = true;
  ballEl.classList.add("dragging");
  var rect = ballEl.getBoundingClientRect();
  var touch = e.touches[0];
  dragOffsetX = touch.clientX - rect.left;
  dragOffsetY = touch.clientY - rect.top;
  e.preventDefault();
});

document.addEventListener("touchmove", function (e) {
  if (!dragging) return;
  var field = document.querySelector(".field");
  var fieldRect = field.getBoundingClientRect();
  var touch = e.touches[0];
  var x = touch.clientX - fieldRect.left - dragOffsetX;
  var y = touch.clientY - fieldRect.top - dragOffsetY;
  x = Math.max(0, Math.min(x, fieldRect.width - 48));
  y = Math.max(0, Math.min(y, fieldRect.height - 48));
  ballEl.style.left = x + "px";
  ballEl.style.top = y + "px";
  ballEl.style.transform = "none";
});

document.addEventListener("touchend", function () {
  dragging = false;
  ballEl.classList.remove("dragging");
});

updateTimer();
updateScores();

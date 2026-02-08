(function () {
  var timerEl = document.getElementById('timer');
  var ballEl = document.getElementById('ball');
  var scoreBlueEl = document.getElementById('scoreBlue');
  var scoreRedEl = document.getElementById('scoreRed');
  var eventsList = document.getElementById('eventsList');

  var minutes = 0;
  var seconds = 54;
  var blueScore = 0;
  var redScore = 2;

  function padTime(m, s) {
    return m + ':' + (s < 10 ? '0' + s : s);
  }

  setInterval(function () {
    seconds++;
    if (seconds >= 60) {
      seconds = 0;
      minutes++;
    }
    timerEl.textContent = padTime(minutes, seconds);
  }, 1000);

  var ballPositions = [
    { left: '48%', top: '40%' },
    { left: '35%', top: '55%' },
    { left: '60%', top: '30%' },
    { left: '75%', top: '50%' },
    { left: '20%', top: '45%' },
    { left: '55%', top: '65%' },
    { left: '40%', top: '25%' },
    { left: '65%', top: '45%' },
    { left: '30%', top: '60%' },
    { left: '50%', top: '50%' },
  ];
  var ballIdx = 0;

  function moveBall() {
    ballIdx = (ballIdx + 1) % ballPositions.length;
    var pos = ballPositions[ballIdx];
    ballEl.style.left = pos.left;
    ballEl.style.top = pos.top;
  }

  setInterval(moveBall, 3000);

  var bluePlayers = document.querySelectorAll('.player--blue');
  var redPlayers = document.querySelectorAll('.player--red');

  var blueZones = [
    [{ x: 12, y: 65 }, { x: 18, y: 50 }, { x: 10, y: 75 }],
    [{ x: 25, y: 35 }, { x: 30, y: 45 }, { x: 22, y: 28 }],
    [{ x: 35, y: 55 }, { x: 40, y: 42 }, { x: 32, y: 60 }],
  ];
  var redZones = [
    [{ x: 62, y: 28 }, { x: 68, y: 35 }, { x: 58, y: 22 }],
    [{ x: 70, y: 55 }, { x: 75, y: 48 }, { x: 65, y: 62 }],
    [{ x: 55, y: 70 }, { x: 60, y: 75 }, { x: 52, y: 65 }],
  ];

  function movePlayersInZone(players, zones) {
    players.forEach(function (p, i) {
      var zone = zones[i];
      var pos = zone[Math.floor(Math.random() * zone.length)];
      p.style.setProperty('--x', pos.x + '%');
      p.style.setProperty('--y', pos.y + '%');
    });
  }

  setInterval(function () {
    movePlayersInZone(bluePlayers, blueZones);
    movePlayersInZone(redPlayers, redZones);
  }, 4000);

  var tooltipEvents = [
    { time: "52'", icon: '🔄', detail: '<strong>Torres</strong> — Substitution (Azure FC)' },
    { time: "58'", icon: '⚽', detail: '<strong>Kante</strong> — Goal! (Azure FC)', goal: 'blue' },
    { time: "67'", icon: '🟨', detail: '<strong>Muller</strong> — Yellow Card (Crimson Utd)' },
    { time: "73'", icon: '⚽', detail: '<strong>Torres</strong> — Goal! (Azure FC)', goal: 'blue' },
    { time: "81'", icon: '🟥', detail: '<strong>Silva</strong> — Red Card (Azure FC)' },
    { time: "88'", icon: '⚽', detail: '<strong>Rashid</strong> — Goal! (Crimson Utd)', goal: 'red' },
  ];
  var nextEvent = 0;

  function addEvent() {
    if (nextEvent >= tooltipEvents.length) return;
    var ev = tooltipEvents[nextEvent];
    nextEvent++;

    var li = document.createElement('li');
    li.className = 'event' + (ev.goal ? ' event--goal' : ' event--card');
    li.innerHTML =
      '<span class="event-time">' + ev.time + '</span>' +
      '<span class="event-icon">' + ev.icon + '</span>' +
      '<span class="event-detail">' + ev.detail + '</span>';
    eventsList.appendChild(li);

    var panel = document.getElementById('eventsPanel');
    panel.scrollTop = panel.scrollHeight;

    if (ev.goal === 'blue') {
      blueScore++;
      animateScore(scoreBlueEl, blueScore);
    } else if (ev.goal === 'red') {
      redScore++;
      animateScore(scoreRedEl, redScore);
    }
  }

  function animateScore(el, val) {
    el.style.transition = 'transform 0.3s ease';
    el.style.transform = 'scale(1.6)';
    el.textContent = val;
    setTimeout(function () {
      el.style.transform = 'scale(1)';
    }, 400);
  }

  setInterval(addEvent, 8000);

  document.querySelectorAll('.player').forEach(function (p) {
    p.addEventListener('click', function () {
      var marker = p.querySelector('.player-marker');
      marker.style.transform = 'scale(1.5)';
      marker.style.boxShadow = '0 0 30px rgba(255,255,255,0.5)';
      setTimeout(function () {
        marker.style.transform = '';
        marker.style.boxShadow = '';
      }, 600);
    });
  });
})();

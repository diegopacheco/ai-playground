import http from 'k6/http';
import { check, sleep } from 'k6';

const BASE_URL = 'http://localhost:3000';

export const options = {
    stages: [
        { duration: '10s', target: 10 },
        { duration: '30s', target: 10 },
        { duration: '5s', target: 0 },
    ],
    thresholds: {
        http_req_duration: ['p(95)<500'],
        http_req_failed: ['rate<0.01'],
    },
};

function createPlayer(vuId) {
    const name = `player_${vuId}_${Date.now()}`;
    const res = http.post(`${BASE_URL}/api/players`, JSON.stringify({ name }), {
        headers: { 'Content-Type': 'application/json' },
    });
    check(res, { 'player created': (r) => r.status === 201 });
    return res.json().id;
}

function createGame(playerId) {
    const res = http.post(`${BASE_URL}/api/games`, JSON.stringify({ player_id: playerId }), {
        headers: { 'Content-Type': 'application/json' },
    });
    check(res, { 'game created': (r) => r.status === 201 });
    return res.json();
}

function flipCard(gameId, position) {
    const res = http.post(`${BASE_URL}/api/games/${gameId}/flip`, JSON.stringify({ position }), {
        headers: { 'Content-Type': 'application/json' },
    });
    check(res, { 'card flipped': (r) => r.status === 200 });
    return res.json();
}

function getGame(gameId) {
    const res = http.get(`${BASE_URL}/api/games/${gameId}`);
    check(res, { 'game retrieved': (r) => r.status === 200 });
    return res.json();
}

function getLeaderboard() {
    const res = http.get(`${BASE_URL}/api/leaderboard`);
    check(res, { 'leaderboard retrieved': (r) => r.status === 200 });
}

function getPlayerStats(playerId) {
    const res = http.get(`${BASE_URL}/api/players/${playerId}/stats`);
    check(res, { 'player stats retrieved': (r) => r.status === 200 });
}

function playGame(gameId) {
    let revealed = {};
    let matched = new Set();

    for (let move = 0; move < 50 && matched.size < 16; move++) {
        let positions = [];
        for (let i = 0; i < 16; i++) {
            if (!matched.has(i)) positions.push(i);
        }

        let first = positions[Math.floor(Math.random() * positions.length)];
        let result1 = flipCard(gameId, first);
        if (!result1 || !result1.game) break;

        if (result1.game.board && result1.game.board[first] && result1.game.board[first].value) {
            revealed[first] = result1.game.board[first].value;
        }

        sleep(0.1);

        let second = -1;
        let firstValue = revealed[first];
        if (firstValue) {
            for (let pos of positions) {
                if (pos !== first && revealed[pos] === firstValue) {
                    second = pos;
                    break;
                }
            }
        }

        if (second === -1) {
            let remaining = positions.filter(p => p !== first);
            second = remaining[Math.floor(Math.random() * remaining.length)];
        }

        let result2 = flipCard(gameId, second);
        if (!result2 || !result2.game) break;

        if (result2.game.board && result2.game.board[second] && result2.game.board[second].value) {
            revealed[second] = result2.game.board[second].value;
        }

        if (result2.matched === true) {
            matched.add(first);
            matched.add(second);
        }

        if (result2.game.status === 'completed') break;

        sleep(0.2);
    }
}

export default function () {
    const playerId = createPlayer(__VU);
    sleep(0.5);

    const game = createGame(playerId);
    sleep(0.3);

    playGame(game.id);

    getGame(game.id);
    sleep(0.2);

    getPlayerStats(playerId);
    sleep(0.2);

    getLeaderboard();
    sleep(1);
}

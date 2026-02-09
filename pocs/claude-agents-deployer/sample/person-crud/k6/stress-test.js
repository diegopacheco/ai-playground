import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

const BASE_URL = 'http://localhost:8080';

const createdPersons = new Counter('created_persons');
const failedRequests = new Rate('failed_requests');
const createDuration = new Trend('create_duration');
const readDuration = new Trend('read_duration');
const adminDuration = new Trend('admin_duration');

export const options = {
    scenarios: {
        create_persons: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                { duration: '10s', target: 10 },
                { duration: '20s', target: 20 },
                { duration: '10s', target: 0 },
            ],
            exec: 'createPersons',
            gracefulRampDown: '5s',
        },
        read_persons: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                { duration: '5s', target: 5 },
                { duration: '20s', target: 30 },
                { duration: '10s', target: 50 },
                { duration: '5s', target: 0 },
            ],
            exec: 'readPersons',
            startTime: '10s',
            gracefulRampDown: '5s',
        },
        admin_views: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                { duration: '5s', target: 5 },
                { duration: '15s', target: 15 },
                { duration: '5s', target: 0 },
            ],
            exec: 'checkAdminViews',
            startTime: '15s',
            gracefulRampDown: '5s',
        },
        mixed_workload: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                { duration: '10s', target: 10 },
                { duration: '20s', target: 40 },
                { duration: '10s', target: 0 },
            ],
            exec: 'mixedWorkload',
            startTime: '5s',
            gracefulRampDown: '5s',
        },
    },
    thresholds: {
        http_req_duration: ['p(95)<500'],
        'create_duration': ['p(95)<500'],
        'read_duration': ['p(95)<500'],
        'admin_duration': ['p(95)<500'],
        'failed_requests': ['rate<0.1'],
    },
};

function randomName() {
    const names = [
        'Alice', 'Bob', 'Charlie', 'Diana', 'Edward',
        'Fiona', 'George', 'Hannah', 'Ivan', 'Julia',
        'Kevin', 'Laura', 'Mike', 'Nancy', 'Oscar',
        'Paula', 'Quinn', 'Rachel', 'Steve', 'Tina',
    ];
    const suffix = Math.floor(Math.random() * 100000);
    return names[Math.floor(Math.random() * names.length)] + suffix;
}

function randomAge() {
    return Math.floor(Math.random() * 60) + 18;
}

export function createPersons() {
    const name = randomName();
    const payload = JSON.stringify({
        name: name,
        email: `${name.toLowerCase()}@test.com`,
        age: randomAge(),
    });

    const params = {
        headers: { 'Content-Type': 'application/json' },
    };

    const res = http.post(`${BASE_URL}/persons`, payload, params);
    createDuration.add(res.timings.duration);

    const success = check(res, {
        'create status is 201': (r) => r.status === 201,
        'create has id': (r) => JSON.parse(r.body).id !== undefined,
    });

    if (success) {
        createdPersons.add(1);
    } else {
        failedRequests.add(1);
    }

    sleep(0.1);
}

export function readPersons() {
    const listRes = http.get(`${BASE_URL}/persons`);
    readDuration.add(listRes.timings.duration);

    check(listRes, {
        'list status is 200': (r) => r.status === 200,
        'list returns array': (r) => Array.isArray(JSON.parse(r.body)),
    });

    let persons = [];
    try {
        persons = JSON.parse(listRes.body);
    } catch (e) {
        failedRequests.add(1);
        return;
    }

    if (persons.length > 0) {
        const randomPerson = persons[Math.floor(Math.random() * persons.length)];
        const detailRes = http.get(`${BASE_URL}/persons/${randomPerson.id}`);
        readDuration.add(detailRes.timings.duration);

        check(detailRes, {
            'detail status is 200': (r) => r.status === 200,
            'detail has name': (r) => JSON.parse(r.body).name !== undefined,
        });
    }

    sleep(0.2);
}

export function checkAdminViews() {
    const allViewsRes = http.get(`${BASE_URL}/admin/views`);
    adminDuration.add(allViewsRes.timings.duration);

    check(allViewsRes, {
        'admin views status is 200': (r) => r.status === 200,
        'admin views returns array': (r) => Array.isArray(JSON.parse(r.body)),
    });

    let views = [];
    try {
        views = JSON.parse(allViewsRes.body);
    } catch (e) {
        failedRequests.add(1);
        return;
    }

    if (views.length > 0) {
        const randomView = views[Math.floor(Math.random() * views.length)];
        const singleViewRes = http.get(`${BASE_URL}/admin/views/${randomView.id}`);
        adminDuration.add(singleViewRes.timings.duration);

        check(singleViewRes, {
            'single view status is 200': (r) => r.status === 200,
            'single view has view_count': (r) => JSON.parse(r.body).view_count !== undefined,
        });
    }

    sleep(0.3);
}

export function mixedWorkload() {
    const action = Math.random();

    if (action < 0.3) {
        const name = randomName();
        const payload = JSON.stringify({
            name: name,
            email: `${name.toLowerCase()}@test.com`,
            age: randomAge(),
        });
        const params = { headers: { 'Content-Type': 'application/json' } };
        const res = http.post(`${BASE_URL}/persons`, payload, params);
        check(res, { 'mixed create 201': (r) => r.status === 201 });
    } else if (action < 0.6) {
        const listRes = http.get(`${BASE_URL}/persons`);
        check(listRes, { 'mixed list 200': (r) => r.status === 200 });

        let persons = [];
        try {
            persons = JSON.parse(listRes.body);
        } catch (e) {
            return;
        }

        if (persons.length > 0) {
            const p = persons[Math.floor(Math.random() * persons.length)];
            const detailRes = http.get(`${BASE_URL}/persons/${p.id}`);
            check(detailRes, { 'mixed detail 200': (r) => r.status === 200 });
        }
    } else if (action < 0.8) {
        const listRes = http.get(`${BASE_URL}/persons`);
        let persons = [];
        try {
            persons = JSON.parse(listRes.body);
        } catch (e) {
            return;
        }

        if (persons.length > 0) {
            const p = persons[Math.floor(Math.random() * persons.length)];
            const updatedName = randomName();
            const payload = JSON.stringify({
                name: updatedName,
                email: `${updatedName.toLowerCase()}@test.com`,
                age: randomAge(),
            });
            const params = { headers: { 'Content-Type': 'application/json' } };
            const res = http.put(`${BASE_URL}/persons/${p.id}`, payload, params);
            check(res, { 'mixed update 200': (r) => r.status === 200 });
        }
    } else if (action < 0.9) {
        const res = http.get(`${BASE_URL}/admin/views`);
        check(res, { 'mixed admin views 200': (r) => r.status === 200 });
    } else {
        const listRes = http.get(`${BASE_URL}/persons`);
        let persons = [];
        try {
            persons = JSON.parse(listRes.body);
        } catch (e) {
            return;
        }

        if (persons.length > 0) {
            const p = persons[Math.floor(Math.random() * persons.length)];
            const res = http.del(`${BASE_URL}/persons/${p.id}`);
            check(res, {
                'mixed delete success': (r) => r.status === 200 || r.status === 404,
            });
        }
    }

    sleep(0.1);
}

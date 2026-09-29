import { test } from 'node:test';
import assert from 'node:assert/strict';
import { placeOf, toLocal, toLatLon } from '../public/js/core/geo.js';
import { loadGeo } from './geo-fixture.js';

const at = (id, lat, lon) => {
  const geo = loadGeo(id);
  const [x, z] = toLocal(placeOf(id), lat, lon);
  return geo.elevation(x, z);
};

test('local meters and lat/lon convert back and forth, so landmarks land on their real spot', () => {
  const p = placeOf('yosemite');
  const [lat, lon] = toLatLon(p, 1234, -567);
  const [x, z] = toLocal(p, lat, lon);
  assert.ok(Math.abs(x - 1234) < 0.01 && Math.abs(z + 567) < 0.01);
});

test('Yosemite is the real valley: floor near 1200 m, Half Dome summit near 2690 m, El Capitan rim above 2100 m', () => {
  assert.ok(Math.abs(at('yosemite', 37.739, -119.605) - 1210) < 40);
  assert.ok(Math.abs(at('yosemite', 37.74604, -119.53294) - 2693) < 120);
  assert.ok(at('yosemite', 37.7422, -119.6358) > 2100);
});

test('Lake Tahoe is the real lake: water surface at 1897 m and Mount Tallac rising above 2800 m', () => {
  assert.ok(Math.abs(at('tahoe', 38.96, -120.03) - 1897.5) < 1);
  assert.ok(at('tahoe', 38.9064, -120.0990) > 2800);
});

test('San Francisco is the real bay: the Golden Gate strait is under sea level and Crissy Field is a few meters above it', () => {
  assert.ok(at('sf', 37.8197, -122.4786) < 0);
  assert.ok(Math.abs(at('sf', 37.8042, -122.4597) - 4) < 4);
});

test('Los Angeles is the real Griffith Park: Mount Lee under the Hollywood Sign stands well above the flats', () => {
  assert.ok(at('la', 34.1341, -118.3216) - at('la', 34.142, -118.286) > 250);
});

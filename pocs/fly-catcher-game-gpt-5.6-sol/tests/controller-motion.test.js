const assert = require('node:assert')
const fs = require('node:fs')
const vm = require('node:vm')

let now = 20
let motionHandler
const requests = []
const completions = []
const elements = new Map()

function element(id) {
  if (!elements.has(id)) {
    elements.set(id, {
      disabled: false,
      style: {},
      textContent: '',
      classList: { toggle() {} },
      addEventListener(name, handler) {
        this[name] = handler
      }
    })
  }
  return elements.get(id)
}

const context = {
  URLSearchParams,
  location: { search: '?token=test' },
  performance: { now: () => now },
  navigator: {},
  DeviceMotionEvent: function DeviceMotionEvent() {},
  document: {
    visibilityState: 'visible',
    getElementById: element,
    addEventListener() {}
  },
  window: {
    isSecureContext: true,
    addEventListener(name, handler) {
      if (name === 'devicemotion') motionHandler = handler
    }
  },
  fetch(url, options) {
    requests.push(JSON.parse(options.body))
    return new Promise((resolve) => completions.push(() => resolve({ ok: true })))
  }
}

vm.createContext(context)
vm.runInContext(fs.readFileSync('public/controller.js', 'utf8'), context)

async function run() {
  const enabled = element('enable').click()
  assert.equal(requests.length, 1)
  completions.shift()()
  await enabled

  motionHandler({ accelerationIncludingGravity: { x: 1, y: 0, z: 9.80665 } })
  now = 37
  motionHandler({ accelerationIncludingGravity: { x: 2, y: 0, z: 9.80665 } })
  now = 54
  motionHandler({ accelerationIncludingGravity: { x: 3, y: 0, z: 9.80665 } })

  assert.equal(requests.length, 2)
  completions.shift()()
  await new Promise(setImmediate)

  assert.equal(requests.length, 3)
  assert.equal(requests[2].ax, 3 / 9.80665)
  completions.shift()()
  process.stdout.write('Latest controller motion survives network backpressure\n')
}

run().catch((error) => {
  process.stderr.write(`${error.stack}\n`)
  process.exit(1)
})

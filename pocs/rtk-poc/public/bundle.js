var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// https://esm.sh/react-dom@18.3.1/denonext/react-dom.mjs
var react_dom_exports = {};
__export(react_dom_exports, {
  __SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED: () => Pf,
  createPortal: () => Lf,
  createRoot: () => Tf,
  default: () => Af,
  findDOMNode: () => Mf,
  flushSync: () => Df,
  hydrate: () => Of,
  hydrateRoot: () => Rf,
  render: () => Ff,
  unmountComponentAtNode: () => If,
  unstable_batchedUpdates: () => jf,
  unstable_renderSubtreeIntoContainer: () => Uf,
  version: () => Vf
});

// https://esm.sh/react@18.3.1/denonext/react.mjs
var react_exports = {};
__export(react_exports, {
  Children: () => le,
  Component: () => ae,
  Fragment: () => pe,
  Profiler: () => ye,
  PureComponent: () => de,
  StrictMode: () => _e,
  Suspense: () => me,
  __SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED: () => he,
  act: () => ve,
  cloneElement: () => Se,
  createContext: () => Ee,
  createElement: () => Re,
  createFactory: () => Ce,
  createRef: () => ke,
  default: () => We,
  forwardRef: () => we,
  isValidElement: () => be,
  lazy: () => $e,
  memo: () => je,
  startTransition: () => xe,
  unstable_act: () => Oe,
  useCallback: () => Ie,
  useContext: () => ge,
  useDebugValue: () => Pe,
  useDeferredValue: () => Te,
  useEffect: () => De,
  useId: () => Ve,
  useImperativeHandle: () => Le,
  useInsertionEffect: () => Ne,
  useLayoutEffect: () => Fe,
  useMemo: () => Ue,
  useReducer: () => qe,
  useRef: () => Ae,
  useState: () => Me,
  useSyncExternalStore: () => ze,
  useTransition: () => Be,
  version: () => He
});
var U = Object.create;
var k = Object.defineProperty;
var q = Object.getOwnPropertyDescriptor;
var A = Object.getOwnPropertyNames;
var M = Object.getPrototypeOf;
var z = Object.prototype.hasOwnProperty;
var w = (e, t) => () => (t || e((t = { exports: {} }).exports, t), t.exports);
var B = (e, t, n2, u2) => {
  if (t && typeof t == "object" || typeof t == "function") for (let o of A(t)) !z.call(e, o) && o !== n2 && k(e, o, { get: () => t[o], enumerable: !(u2 = q(t, o)) || u2.enumerable });
  return e;
};
var H = (e, t, n2) => (n2 = e != null ? U(M(e)) : {}, B(t || !e || !e.__esModule ? k(n2, "default", { value: e, enumerable: true }) : n2, e));
var L = w((r) => {
  "use strict";
  var y3 = Symbol.for("react.element"), W = Symbol.for("react.portal"), Y = Symbol.for("react.fragment"), G = Symbol.for("react.strict_mode"), J = Symbol.for("react.profiler"), K2 = Symbol.for("react.provider"), Q = Symbol.for("react.context"), X2 = Symbol.for("react.forward_ref"), Z2 = Symbol.for("react.suspense"), ee2 = Symbol.for("react.memo"), te = Symbol.for("react.lazy"), b = Symbol.iterator;
  function re(e) {
    return e === null || typeof e != "object" ? null : (e = b && e[b] || e["@@iterator"], typeof e == "function" ? e : null);
  }
  var x4 = { isMounted: function() {
    return false;
  }, enqueueForceUpdate: function() {
  }, enqueueReplaceState: function() {
  }, enqueueSetState: function() {
  } }, O3 = Object.assign, I2 = {};
  function p(e, t, n2) {
    this.props = e, this.context = t, this.refs = I2, this.updater = n2 || x4;
  }
  p.prototype.isReactComponent = {};
  p.prototype.setState = function(e, t) {
    if (typeof e != "object" && typeof e != "function" && e != null) throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");
    this.updater.enqueueSetState(this, e, t, "setState");
  };
  p.prototype.forceUpdate = function(e) {
    this.updater.enqueueForceUpdate(this, e, "forceUpdate");
  };
  function g2() {
  }
  g2.prototype = p.prototype;
  function S2(e, t, n2) {
    this.props = e, this.context = t, this.refs = I2, this.updater = n2 || x4;
  }
  var E3 = S2.prototype = new g2();
  E3.constructor = S2;
  O3(E3, p.prototype);
  E3.isPureReactComponent = true;
  var $2 = Array.isArray, P = Object.prototype.hasOwnProperty, R2 = { current: null }, T2 = { key: true, ref: true, __self: true, __source: true };
  function D2(e, t, n2) {
    var u2, o = {}, s = null, f3 = null;
    if (t != null) for (u2 in t.ref !== void 0 && (f3 = t.ref), t.key !== void 0 && (s = "" + t.key), t) P.call(t, u2) && !T2.hasOwnProperty(u2) && (o[u2] = t[u2]);
    var i2 = arguments.length - 2;
    if (i2 === 1) o.children = n2;
    else if (1 < i2) {
      for (var c3 = Array(i2), a2 = 0; a2 < i2; a2++) c3[a2] = arguments[a2 + 2];
      o.children = c3;
    }
    if (e && e.defaultProps) for (u2 in i2 = e.defaultProps, i2) o[u2] === void 0 && (o[u2] = i2[u2]);
    return { $$typeof: y3, type: e, key: s, ref: f3, props: o, _owner: R2.current };
  }
  function ne2(e, t) {
    return { $$typeof: y3, type: e.type, key: t, ref: e.ref, props: e.props, _owner: e._owner };
  }
  function C(e) {
    return typeof e == "object" && e !== null && e.$$typeof === y3;
  }
  function oe2(e) {
    var t = { "=": "=0", ":": "=2" };
    return "$" + e.replace(/[=:]/g, function(n2) {
      return t[n2];
    });
  }
  var j2 = /\/+/g;
  function v2(e, t) {
    return typeof e == "object" && e !== null && e.key != null ? oe2("" + e.key) : t.toString(36);
  }
  function _2(e, t, n2, u2, o) {
    var s = typeof e;
    (s === "undefined" || s === "boolean") && (e = null);
    var f3 = false;
    if (e === null) f3 = true;
    else switch (s) {
      case "string":
      case "number":
        f3 = true;
        break;
      case "object":
        switch (e.$$typeof) {
          case y3:
          case W:
            f3 = true;
        }
    }
    if (f3) return f3 = e, o = o(f3), e = u2 === "" ? "." + v2(f3, 0) : u2, $2(o) ? (n2 = "", e != null && (n2 = e.replace(j2, "$&/") + "/"), _2(o, t, n2, "", function(a2) {
      return a2;
    })) : o != null && (C(o) && (o = ne2(o, n2 + (!o.key || f3 && f3.key === o.key ? "" : ("" + o.key).replace(j2, "$&/") + "/") + e)), t.push(o)), 1;
    if (f3 = 0, u2 = u2 === "" ? "." : u2 + ":", $2(e)) for (var i2 = 0; i2 < e.length; i2++) {
      s = e[i2];
      var c3 = u2 + v2(s, i2);
      f3 += _2(s, t, n2, c3, o);
    }
    else if (c3 = re(e), typeof c3 == "function") for (e = c3.call(e), i2 = 0; !(s = e.next()).done; ) s = s.value, c3 = u2 + v2(s, i2++), f3 += _2(s, t, n2, c3, o);
    else if (s === "object") throw t = String(e), Error("Objects are not valid as a React child (found: " + (t === "[object Object]" ? "object with keys {" + Object.keys(e).join(", ") + "}" : t) + "). If you meant to render a collection of children, use an array instead.");
    return f3;
  }
  function d3(e, t, n2) {
    if (e == null) return e;
    var u2 = [], o = 0;
    return _2(e, u2, "", "", function(s) {
      return t.call(n2, s, o++);
    }), u2;
  }
  function ue2(e) {
    if (e._status === -1) {
      var t = e._result;
      t = t(), t.then(function(n2) {
        (e._status === 0 || e._status === -1) && (e._status = 1, e._result = n2);
      }, function(n2) {
        (e._status === 0 || e._status === -1) && (e._status = 2, e._result = n2);
      }), e._status === -1 && (e._status = 0, e._result = t);
    }
    if (e._status === 1) return e._result.default;
    throw e._result;
  }
  var l3 = { current: null }, m2 = { transition: null }, ce2 = { ReactCurrentDispatcher: l3, ReactCurrentBatchConfig: m2, ReactCurrentOwner: R2 };
  function V2() {
    throw Error("act(...) is not supported in production builds of React.");
  }
  r.Children = { map: d3, forEach: function(e, t, n2) {
    d3(e, function() {
      t.apply(this, arguments);
    }, n2);
  }, count: function(e) {
    var t = 0;
    return d3(e, function() {
      t++;
    }), t;
  }, toArray: function(e) {
    return d3(e, function(t) {
      return t;
    }) || [];
  }, only: function(e) {
    if (!C(e)) throw Error("React.Children.only expected to receive a single React element child.");
    return e;
  } };
  r.Component = p;
  r.Fragment = Y;
  r.Profiler = J;
  r.PureComponent = S2;
  r.StrictMode = G;
  r.Suspense = Z2;
  r.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = ce2;
  r.act = V2;
  r.cloneElement = function(e, t, n2) {
    if (e == null) throw Error("React.cloneElement(...): The argument must be a React element, but you passed " + e + ".");
    var u2 = O3({}, e.props), o = e.key, s = e.ref, f3 = e._owner;
    if (t != null) {
      if (t.ref !== void 0 && (s = t.ref, f3 = R2.current), t.key !== void 0 && (o = "" + t.key), e.type && e.type.defaultProps) var i2 = e.type.defaultProps;
      for (c3 in t) P.call(t, c3) && !T2.hasOwnProperty(c3) && (u2[c3] = t[c3] === void 0 && i2 !== void 0 ? i2[c3] : t[c3]);
    }
    var c3 = arguments.length - 2;
    if (c3 === 1) u2.children = n2;
    else if (1 < c3) {
      i2 = Array(c3);
      for (var a2 = 0; a2 < c3; a2++) i2[a2] = arguments[a2 + 2];
      u2.children = i2;
    }
    return { $$typeof: y3, type: e.type, key: o, ref: s, props: u2, _owner: f3 };
  };
  r.createContext = function(e) {
    return e = { $$typeof: Q, _currentValue: e, _currentValue2: e, _threadCount: 0, Provider: null, Consumer: null, _defaultValue: null, _globalName: null }, e.Provider = { $$typeof: K2, _context: e }, e.Consumer = e;
  };
  r.createElement = D2;
  r.createFactory = function(e) {
    var t = D2.bind(null, e);
    return t.type = e, t;
  };
  r.createRef = function() {
    return { current: null };
  };
  r.forwardRef = function(e) {
    return { $$typeof: X2, render: e };
  };
  r.isValidElement = C;
  r.lazy = function(e) {
    return { $$typeof: te, _payload: { _status: -1, _result: e }, _init: ue2 };
  };
  r.memo = function(e, t) {
    return { $$typeof: ee2, type: e, compare: t === void 0 ? null : t };
  };
  r.startTransition = function(e) {
    var t = m2.transition;
    m2.transition = {};
    try {
      e();
    } finally {
      m2.transition = t;
    }
  };
  r.unstable_act = V2;
  r.useCallback = function(e, t) {
    return l3.current.useCallback(e, t);
  };
  r.useContext = function(e) {
    return l3.current.useContext(e);
  };
  r.useDebugValue = function() {
  };
  r.useDeferredValue = function(e) {
    return l3.current.useDeferredValue(e);
  };
  r.useEffect = function(e, t) {
    return l3.current.useEffect(e, t);
  };
  r.useId = function() {
    return l3.current.useId();
  };
  r.useImperativeHandle = function(e, t, n2) {
    return l3.current.useImperativeHandle(e, t, n2);
  };
  r.useInsertionEffect = function(e, t) {
    return l3.current.useInsertionEffect(e, t);
  };
  r.useLayoutEffect = function(e, t) {
    return l3.current.useLayoutEffect(e, t);
  };
  r.useMemo = function(e, t) {
    return l3.current.useMemo(e, t);
  };
  r.useReducer = function(e, t, n2) {
    return l3.current.useReducer(e, t, n2);
  };
  r.useRef = function(e) {
    return l3.current.useRef(e);
  };
  r.useState = function(e) {
    return l3.current.useState(e);
  };
  r.useSyncExternalStore = function(e, t, n2) {
    return l3.current.useSyncExternalStore(e, t, n2);
  };
  r.useTransition = function() {
    return l3.current.useTransition();
  };
  r.version = "18.3.1";
});
var F = w((fe2, N) => {
  "use strict";
  N.exports = L();
});
var h = H(F());
var { Children: le, Component: ae, Fragment: pe, Profiler: ye, PureComponent: de, StrictMode: _e, Suspense: me, __SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED: he, act: ve, cloneElement: Se, createContext: Ee, createElement: Re, createFactory: Ce, createRef: ke, forwardRef: we, isValidElement: be, lazy: $e, memo: je, startTransition: xe, unstable_act: Oe, useCallback: Ie, useContext: ge, useDebugValue: Pe, useDeferredValue: Te, useEffect: De, useId: Ve, useImperativeHandle: Le, useInsertionEffect: Ne, useLayoutEffect: Fe, useMemo: Ue, useReducer: qe, useRef: Ae, useState: Me, useSyncExternalStore: ze, useTransition: Be, version: He } = h;
var We = h.default ?? h;

// https://esm.sh/scheduler@0.23.2?target=denonext
var scheduler_0_23_exports = {};
__export(scheduler_0_23_exports, {
  default: () => Ie2,
  unstable_IdlePriority: () => ae2,
  unstable_ImmediatePriority: () => oe,
  unstable_LowPriority: () => se,
  unstable_NormalPriority: () => ce,
  unstable_Profiling: () => fe,
  unstable_UserBlockingPriority: () => be2,
  unstable_cancelCallback: () => _e2,
  unstable_continueExecution: () => pe2,
  unstable_forceFrameRate: () => ve2,
  unstable_getCurrentPriorityLevel: () => de2,
  unstable_getFirstCallbackNode: () => ye2,
  unstable_next: () => me2,
  unstable_now: () => ue,
  unstable_pauseExecution: () => ge2,
  unstable_requestPaint: () => he2,
  unstable_runWithPriority: () => ke2,
  unstable_scheduleCallback: () => Pe2,
  unstable_shouldYield: () => we2,
  unstable_wrapCallback: () => xe2
});

// https://esm.sh/scheduler@0.23.2/denonext/scheduler.mjs
var __setImmediate$ = (cb, ...args) => ({ $t: setTimeout(cb, 0, ...args), [Symbol.dispose]() {
  clearTimeout(this.t);
} });
var V = Object.create;
var B2 = Object.defineProperty;
var U2 = Object.getOwnPropertyDescriptor;
var X = Object.getOwnPropertyNames;
var Z = Object.getPrototypeOf;
var $ = Object.prototype.hasOwnProperty;
var D = (e, n2) => () => (n2 || e((n2 = { exports: {} }).exports, n2), n2.exports);
var ee = (e, n2, t, l3) => {
  if (n2 && typeof n2 == "object" || typeof n2 == "function") for (let i2 of X(n2)) !$.call(e, i2) && i2 !== t && B2(e, i2, { get: () => n2[i2], enumerable: !(l3 = U2(n2, i2)) || l3.enumerable });
  return e;
};
var ne = (e, n2, t) => (t = e != null ? V(Z(e)) : {}, ee(n2 || !e || !e.__esModule ? B2(t, "default", { value: e, enumerable: true }) : t, e));
var K = D((r) => {
  "use strict";
  function L3(e, n2) {
    var t = e.length;
    e.push(n2);
    e: for (; 0 < t; ) {
      var l3 = t - 1 >>> 1, i2 = e[l3];
      if (0 < g2(i2, n2)) e[l3] = n2, e[t] = i2, t = l3;
      else break e;
    }
  }
  function o(e) {
    return e.length === 0 ? null : e[0];
  }
  function k3(e) {
    if (e.length === 0) return null;
    var n2 = e[0], t = e.pop();
    if (t !== n2) {
      e[0] = t;
      e: for (var l3 = 0, i2 = e.length, y3 = i2 >>> 1; l3 < y3; ) {
        var f3 = 2 * (l3 + 1) - 1, I2 = e[f3], b = f3 + 1, m2 = e[b];
        if (0 > g2(I2, t)) b < i2 && 0 > g2(m2, I2) ? (e[l3] = m2, e[b] = t, l3 = b) : (e[l3] = I2, e[f3] = t, l3 = f3);
        else if (b < i2 && 0 > g2(m2, t)) e[l3] = m2, e[b] = t, l3 = b;
        else break e;
      }
    }
    return n2;
  }
  function g2(e, n2) {
    var t = e.sortIndex - n2.sortIndex;
    return t !== 0 ? t : e.id - n2.id;
  }
  typeof performance == "object" && typeof performance.now == "function" ? (q2 = performance, r.unstable_now = function() {
    return q2.now();
  }) : (C = Date, O3 = C.now(), r.unstable_now = function() {
    return C.now() - O3;
  });
  var q2, C, O3, s = [], c3 = [], te = 1, a2 = null, u2 = 3, P = false, _2 = false, v2 = false, z2 = typeof setTimeout == "function" ? setTimeout : null, A2 = typeof clearTimeout == "function" ? clearTimeout : null, W = typeof __setImmediate$ < "u" ? __setImmediate$ : null;
  typeof navigator < "u" && navigator.scheduling !== void 0 && navigator.scheduling.isInputPending !== void 0 && navigator.scheduling.isInputPending.bind(navigator.scheduling);
  function N(e) {
    for (var n2 = o(c3); n2 !== null; ) {
      if (n2.callback === null) k3(c3);
      else if (n2.startTime <= e) k3(c3), n2.sortIndex = n2.expirationTime, L3(s, n2);
      else break;
      n2 = o(c3);
    }
  }
  function j2(e) {
    if (v2 = false, N(e), !_2) if (o(s) !== null) _2 = true, M2(F3);
    else {
      var n2 = o(c3);
      n2 !== null && R2(j2, n2.startTime - e);
    }
  }
  function F3(e, n2) {
    _2 = false, v2 && (v2 = false, A2(d3), d3 = -1), P = true;
    var t = u2;
    try {
      for (N(n2), a2 = o(s); a2 !== null && (!(a2.expirationTime > n2) || e && !J()); ) {
        var l3 = a2.callback;
        if (typeof l3 == "function") {
          a2.callback = null, u2 = a2.priorityLevel;
          var i2 = l3(a2.expirationTime <= n2);
          n2 = r.unstable_now(), typeof i2 == "function" ? a2.callback = i2 : a2 === o(s) && k3(s), N(n2);
        } else k3(s);
        a2 = o(s);
      }
      if (a2 !== null) var y3 = true;
      else {
        var f3 = o(c3);
        f3 !== null && R2(j2, f3.startTime - n2), y3 = false;
      }
      return y3;
    } finally {
      a2 = null, u2 = t, P = false;
    }
  }
  var w2 = false, h3 = null, d3 = -1, G = 5, H2 = -1;
  function J() {
    return !(r.unstable_now() - H2 < G);
  }
  function E3() {
    if (h3 !== null) {
      var e = r.unstable_now();
      H2 = e;
      var n2 = true;
      try {
        n2 = h3(true, e);
      } finally {
        n2 ? p() : (w2 = false, h3 = null);
      }
    } else w2 = false;
  }
  var p;
  typeof W == "function" ? p = function() {
    W(E3);
  } : typeof MessageChannel < "u" ? (T2 = new MessageChannel(), Y = T2.port2, T2.port1.onmessage = E3, p = function() {
    Y.postMessage(null);
  }) : p = function() {
    z2(E3, 0);
  };
  var T2, Y;
  function M2(e) {
    h3 = e, w2 || (w2 = true, p());
  }
  function R2(e, n2) {
    d3 = z2(function() {
      e(r.unstable_now());
    }, n2);
  }
  r.unstable_IdlePriority = 5;
  r.unstable_ImmediatePriority = 1;
  r.unstable_LowPriority = 4;
  r.unstable_NormalPriority = 3;
  r.unstable_Profiling = null;
  r.unstable_UserBlockingPriority = 2;
  r.unstable_cancelCallback = function(e) {
    e.callback = null;
  };
  r.unstable_continueExecution = function() {
    _2 || P || (_2 = true, M2(F3));
  };
  r.unstable_forceFrameRate = function(e) {
    0 > e || 125 < e ? console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported") : G = 0 < e ? Math.floor(1e3 / e) : 5;
  };
  r.unstable_getCurrentPriorityLevel = function() {
    return u2;
  };
  r.unstable_getFirstCallbackNode = function() {
    return o(s);
  };
  r.unstable_next = function(e) {
    switch (u2) {
      case 1:
      case 2:
      case 3:
        var n2 = 3;
        break;
      default:
        n2 = u2;
    }
    var t = u2;
    u2 = n2;
    try {
      return e();
    } finally {
      u2 = t;
    }
  };
  r.unstable_pauseExecution = function() {
  };
  r.unstable_requestPaint = function() {
  };
  r.unstable_runWithPriority = function(e, n2) {
    switch (e) {
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
        break;
      default:
        e = 3;
    }
    var t = u2;
    u2 = e;
    try {
      return n2();
    } finally {
      u2 = t;
    }
  };
  r.unstable_scheduleCallback = function(e, n2, t) {
    var l3 = r.unstable_now();
    switch (typeof t == "object" && t !== null ? (t = t.delay, t = typeof t == "number" && 0 < t ? l3 + t : l3) : t = l3, e) {
      case 1:
        var i2 = -1;
        break;
      case 2:
        i2 = 250;
        break;
      case 5:
        i2 = 1073741823;
        break;
      case 4:
        i2 = 1e4;
        break;
      default:
        i2 = 5e3;
    }
    return i2 = t + i2, e = { id: te++, callback: n2, priorityLevel: e, startTime: t, expirationTime: i2, sortIndex: -1 }, t > l3 ? (e.sortIndex = t, L3(c3, e), o(s) === null && e === o(c3) && (v2 ? (A2(d3), d3 = -1) : v2 = true, R2(j2, t - l3))) : (e.sortIndex = i2, L3(s, e), _2 || P || (_2 = true, M2(F3))), e;
  };
  r.unstable_shouldYield = J;
  r.unstable_wrapCallback = function(e) {
    var n2 = u2;
    return function() {
      var t = u2;
      u2 = n2;
      try {
        return e.apply(this, arguments);
      } finally {
        u2 = t;
      }
    };
  };
});
var S = D((ie, Q) => {
  "use strict";
  Q.exports = K();
});
var x = ne(S());
var { unstable_now: ue, unstable_IdlePriority: ae2, unstable_ImmediatePriority: oe, unstable_LowPriority: se, unstable_NormalPriority: ce, unstable_Profiling: fe, unstable_UserBlockingPriority: be2, unstable_cancelCallback: _e2, unstable_continueExecution: pe2, unstable_forceFrameRate: ve2, unstable_getCurrentPriorityLevel: de2, unstable_getFirstCallbackNode: ye2, unstable_next: me2, unstable_pauseExecution: ge2, unstable_requestPaint: he2, unstable_runWithPriority: ke2, unstable_scheduleCallback: Pe2, unstable_shouldYield: we2, unstable_wrapCallback: xe2 } = x;
var Ie2 = x.default ?? x;

// https://esm.sh/react-dom@18.3.1/denonext/react-dom.mjs
var require2 = (n2) => {
  const e = (m2) => typeof m2.default < "u" ? m2.default : m2, c3 = (m2) => Object.assign({ __esModule: true }, m2);
  switch (n2) {
    case "react":
      return e(react_exports);
    case "scheduler":
      return e(scheduler_0_23_exports);
    default:
      console.error('module "' + n2 + '" not found');
      return null;
  }
};
var va = Object.create;
var lu = Object.defineProperty;
var ya = Object.getOwnPropertyDescriptor;
var ga = Object.getOwnPropertyNames;
var wa = Object.getPrototypeOf;
var Sa = Object.prototype.hasOwnProperty;
var iu = ((e) => typeof require2 < "u" ? require2 : typeof Proxy < "u" ? new Proxy(e, { get: (n2, t) => (typeof require2 < "u" ? require2 : n2)[t] }) : e)(function(e) {
  if (typeof require2 < "u") return require2.apply(this, arguments);
  throw Error('Dynamic require of "' + e + '" is not supported');
});
var uu = (e, n2) => () => (n2 || e((n2 = { exports: {} }).exports, n2), n2.exports);
var ka = (e, n2, t, r) => {
  if (n2 && typeof n2 == "object" || typeof n2 == "function") for (let l3 of ga(n2)) !Sa.call(e, l3) && l3 !== t && lu(e, l3, { get: () => n2[l3], enumerable: !(r = ya(n2, l3)) || r.enumerable });
  return e;
};
var Ea = (e, n2, t) => (t = e != null ? va(wa(e)) : {}, ka(n2 || !e || !e.__esModule ? lu(t, "default", { value: e, enumerable: true }) : t, e));
var fa = uu((fe2) => {
  "use strict";
  var Ca = iu("react"), ae3 = iu("scheduler");
  function v2(e) {
    for (var n2 = "https://reactjs.org/docs/error-decoder.html?invariant=" + e, t = 1; t < arguments.length; t++) n2 += "&args[]=" + encodeURIComponent(arguments[t]);
    return "Minified React error #" + e + "; visit " + n2 + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  var mo = /* @__PURE__ */ new Set(), gt = {};
  function Sn(e, n2) {
    Hn(e, n2), Hn(e + "Capture", n2);
  }
  function Hn(e, n2) {
    for (gt[e] = n2, e = 0; e < n2.length; e++) mo.add(n2[e]);
  }
  var Fe2 = !(typeof globalThis > "u" || typeof globalThis.document > "u" || typeof globalThis.document.createElement > "u"), El = Object.prototype.hasOwnProperty, xa = /^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/, ou = {}, su = {};
  function Na(e) {
    return El.call(su, e) ? true : El.call(ou, e) ? false : xa.test(e) ? su[e] = true : (ou[e] = true, false);
  }
  function _a(e, n2, t, r) {
    if (t !== null && t.type === 0) return false;
    switch (typeof n2) {
      case "function":
      case "symbol":
        return true;
      case "boolean":
        return r ? false : t !== null ? !t.acceptsBooleans : (e = e.toLowerCase().slice(0, 5), e !== "data-" && e !== "aria-");
      default:
        return false;
    }
  }
  function za(e, n2, t, r) {
    if (n2 === null || typeof n2 > "u" || _a(e, n2, t, r)) return true;
    if (r) return false;
    if (t !== null) switch (t.type) {
      case 3:
        return !n2;
      case 4:
        return n2 === false;
      case 5:
        return isNaN(n2);
      case 6:
        return isNaN(n2) || 1 > n2;
    }
    return false;
  }
  function ee2(e, n2, t, r, l3, i2, u2) {
    this.acceptsBooleans = n2 === 2 || n2 === 3 || n2 === 4, this.attributeName = r, this.attributeNamespace = l3, this.mustUseProperty = t, this.propertyName = e, this.type = n2, this.sanitizeURL = i2, this.removeEmptyString = u2;
  }
  var Y = {};
  "children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e) {
    Y[e] = new ee2(e, 0, false, e, null, false, false);
  });
  [["acceptCharset", "accept-charset"], ["className", "class"], ["htmlFor", "for"], ["httpEquiv", "http-equiv"]].forEach(function(e) {
    var n2 = e[0];
    Y[n2] = new ee2(n2, 1, false, e[1], null, false, false);
  });
  ["contentEditable", "draggable", "spellCheck", "value"].forEach(function(e) {
    Y[e] = new ee2(e, 2, false, e.toLowerCase(), null, false, false);
  });
  ["autoReverse", "externalResourcesRequired", "focusable", "preserveAlpha"].forEach(function(e) {
    Y[e] = new ee2(e, 2, false, e, null, false, false);
  });
  "allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e) {
    Y[e] = new ee2(e, 3, false, e.toLowerCase(), null, false, false);
  });
  ["checked", "multiple", "muted", "selected"].forEach(function(e) {
    Y[e] = new ee2(e, 3, true, e, null, false, false);
  });
  ["capture", "download"].forEach(function(e) {
    Y[e] = new ee2(e, 4, false, e, null, false, false);
  });
  ["cols", "rows", "size", "span"].forEach(function(e) {
    Y[e] = new ee2(e, 6, false, e, null, false, false);
  });
  ["rowSpan", "start"].forEach(function(e) {
    Y[e] = new ee2(e, 5, false, e.toLowerCase(), null, false, false);
  });
  var mi = /[\-:]([a-z])/g;
  function hi(e) {
    return e[1].toUpperCase();
  }
  "accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e) {
    var n2 = e.replace(mi, hi);
    Y[n2] = new ee2(n2, 1, false, e, null, false, false);
  });
  "xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e) {
    var n2 = e.replace(mi, hi);
    Y[n2] = new ee2(n2, 1, false, e, "http://www.w3.org/1999/xlink", false, false);
  });
  ["xml:base", "xml:lang", "xml:space"].forEach(function(e) {
    var n2 = e.replace(mi, hi);
    Y[n2] = new ee2(n2, 1, false, e, "http://www.w3.org/XML/1998/namespace", false, false);
  });
  ["tabIndex", "crossOrigin"].forEach(function(e) {
    Y[e] = new ee2(e, 1, false, e.toLowerCase(), null, false, false);
  });
  Y.xlinkHref = new ee2("xlinkHref", 1, false, "xlink:href", "http://www.w3.org/1999/xlink", true, false);
  ["src", "href", "action", "formAction"].forEach(function(e) {
    Y[e] = new ee2(e, 1, false, e.toLowerCase(), null, true, true);
  });
  function vi(e, n2, t, r) {
    var l3 = Y.hasOwnProperty(n2) ? Y[n2] : null;
    (l3 !== null ? l3.type !== 0 : r || !(2 < n2.length) || n2[0] !== "o" && n2[0] !== "O" || n2[1] !== "n" && n2[1] !== "N") && (za(n2, t, l3, r) && (t = null), r || l3 === null ? Na(n2) && (t === null ? e.removeAttribute(n2) : e.setAttribute(n2, "" + t)) : l3.mustUseProperty ? e[l3.propertyName] = t === null ? l3.type === 3 ? false : "" : t : (n2 = l3.attributeName, r = l3.attributeNamespace, t === null ? e.removeAttribute(n2) : (l3 = l3.type, t = l3 === 3 || l3 === 4 && t === true ? "" : "" + t, r ? e.setAttributeNS(r, n2, t) : e.setAttribute(n2, t))));
  }
  var Ve2 = Ca.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED, Vt = Symbol.for("react.element"), xn = Symbol.for("react.portal"), Nn = Symbol.for("react.fragment"), yi = Symbol.for("react.strict_mode"), Cl = Symbol.for("react.profiler"), ho = Symbol.for("react.provider"), vo = Symbol.for("react.context"), gi = Symbol.for("react.forward_ref"), xl = Symbol.for("react.suspense"), Nl = Symbol.for("react.suspense_list"), wi = Symbol.for("react.memo"), He2 = Symbol.for("react.lazy");
  Symbol.for("react.scope");
  Symbol.for("react.debug_trace_mode");
  var yo = Symbol.for("react.offscreen");
  Symbol.for("react.legacy_hidden");
  Symbol.for("react.cache");
  Symbol.for("react.tracing_marker");
  var au = Symbol.iterator;
  function Jn(e) {
    return e === null || typeof e != "object" ? null : (e = au && e[au] || e["@@iterator"], typeof e == "function" ? e : null);
  }
  var F3 = Object.assign, el;
  function it(e) {
    if (el === void 0) try {
      throw Error();
    } catch (t) {
      var n2 = t.stack.trim().match(/\n( *(at )?)/);
      el = n2 && n2[1] || "";
    }
    return `
` + el + e;
  }
  var nl = false;
  function tl(e, n2) {
    if (!e || nl) return "";
    nl = true;
    var t = Error.prepareStackTrace;
    Error.prepareStackTrace = void 0;
    try {
      if (n2) if (n2 = function() {
        throw Error();
      }, Object.defineProperty(n2.prototype, "props", { set: function() {
        throw Error();
      } }), typeof Reflect == "object" && Reflect.construct) {
        try {
          Reflect.construct(n2, []);
        } catch (d3) {
          var r = d3;
        }
        Reflect.construct(e, [], n2);
      } else {
        try {
          n2.call();
        } catch (d3) {
          r = d3;
        }
        e.call(n2.prototype);
      }
      else {
        try {
          throw Error();
        } catch (d3) {
          r = d3;
        }
        e();
      }
    } catch (d3) {
      if (d3 && r && typeof d3.stack == "string") {
        for (var l3 = d3.stack.split(`
`), i2 = r.stack.split(`
`), u2 = l3.length - 1, o = i2.length - 1; 1 <= u2 && 0 <= o && l3[u2] !== i2[o]; ) o--;
        for (; 1 <= u2 && 0 <= o; u2--, o--) if (l3[u2] !== i2[o]) {
          if (u2 !== 1 || o !== 1) do
            if (u2--, o--, 0 > o || l3[u2] !== i2[o]) {
              var s = `
` + l3[u2].replace(" at new ", " at ");
              return e.displayName && s.includes("<anonymous>") && (s = s.replace("<anonymous>", e.displayName)), s;
            }
          while (1 <= u2 && 0 <= o);
          break;
        }
      }
    } finally {
      nl = false, Error.prepareStackTrace = t;
    }
    return (e = e ? e.displayName || e.name : "") ? it(e) : "";
  }
  function Pa(e) {
    switch (e.tag) {
      case 5:
        return it(e.type);
      case 16:
        return it("Lazy");
      case 13:
        return it("Suspense");
      case 19:
        return it("SuspenseList");
      case 0:
      case 2:
      case 15:
        return e = tl(e.type, false), e;
      case 11:
        return e = tl(e.type.render, false), e;
      case 1:
        return e = tl(e.type, true), e;
      default:
        return "";
    }
  }
  function _l(e) {
    if (e == null) return null;
    if (typeof e == "function") return e.displayName || e.name || null;
    if (typeof e == "string") return e;
    switch (e) {
      case Nn:
        return "Fragment";
      case xn:
        return "Portal";
      case Cl:
        return "Profiler";
      case yi:
        return "StrictMode";
      case xl:
        return "Suspense";
      case Nl:
        return "SuspenseList";
    }
    if (typeof e == "object") switch (e.$$typeof) {
      case vo:
        return (e.displayName || "Context") + ".Consumer";
      case ho:
        return (e._context.displayName || "Context") + ".Provider";
      case gi:
        var n2 = e.render;
        return e = e.displayName, e || (e = n2.displayName || n2.name || "", e = e !== "" ? "ForwardRef(" + e + ")" : "ForwardRef"), e;
      case wi:
        return n2 = e.displayName || null, n2 !== null ? n2 : _l(e.type) || "Memo";
      case He2:
        n2 = e._payload, e = e._init;
        try {
          return _l(e(n2));
        } catch {
        }
    }
    return null;
  }
  function La(e) {
    var n2 = e.type;
    switch (e.tag) {
      case 24:
        return "Cache";
      case 9:
        return (n2.displayName || "Context") + ".Consumer";
      case 10:
        return (n2._context.displayName || "Context") + ".Provider";
      case 18:
        return "DehydratedFragment";
      case 11:
        return e = n2.render, e = e.displayName || e.name || "", n2.displayName || (e !== "" ? "ForwardRef(" + e + ")" : "ForwardRef");
      case 7:
        return "Fragment";
      case 5:
        return n2;
      case 4:
        return "Portal";
      case 3:
        return "Root";
      case 6:
        return "Text";
      case 16:
        return _l(n2);
      case 8:
        return n2 === yi ? "StrictMode" : "Mode";
      case 22:
        return "Offscreen";
      case 12:
        return "Profiler";
      case 21:
        return "Scope";
      case 13:
        return "Suspense";
      case 19:
        return "SuspenseList";
      case 25:
        return "TracingMarker";
      case 1:
      case 0:
      case 17:
      case 2:
      case 14:
      case 15:
        if (typeof n2 == "function") return n2.displayName || n2.name || null;
        if (typeof n2 == "string") return n2;
    }
    return null;
  }
  function tn(e) {
    switch (typeof e) {
      case "boolean":
      case "number":
      case "string":
      case "undefined":
        return e;
      case "object":
        return e;
      default:
        return "";
    }
  }
  function go(e) {
    var n2 = e.type;
    return (e = e.nodeName) && e.toLowerCase() === "input" && (n2 === "checkbox" || n2 === "radio");
  }
  function Ta(e) {
    var n2 = go(e) ? "checked" : "value", t = Object.getOwnPropertyDescriptor(e.constructor.prototype, n2), r = "" + e[n2];
    if (!e.hasOwnProperty(n2) && typeof t < "u" && typeof t.get == "function" && typeof t.set == "function") {
      var l3 = t.get, i2 = t.set;
      return Object.defineProperty(e, n2, { configurable: true, get: function() {
        return l3.call(this);
      }, set: function(u2) {
        r = "" + u2, i2.call(this, u2);
      } }), Object.defineProperty(e, n2, { enumerable: t.enumerable }), { getValue: function() {
        return r;
      }, setValue: function(u2) {
        r = "" + u2;
      }, stopTracking: function() {
        e._valueTracker = null, delete e[n2];
      } };
    }
  }
  function At(e) {
    e._valueTracker || (e._valueTracker = Ta(e));
  }
  function wo(e) {
    if (!e) return false;
    var n2 = e._valueTracker;
    if (!n2) return true;
    var t = n2.getValue(), r = "";
    return e && (r = go(e) ? e.checked ? "true" : "false" : e.value), e = r, e !== t ? (n2.setValue(e), true) : false;
  }
  function mr(e) {
    if (e = e || (typeof document < "u" ? document : void 0), typeof e > "u") return null;
    try {
      return e.activeElement || e.body;
    } catch {
      return e.body;
    }
  }
  function zl(e, n2) {
    var t = n2.checked;
    return F3({}, n2, { defaultChecked: void 0, defaultValue: void 0, value: void 0, checked: t ?? e._wrapperState.initialChecked });
  }
  function cu(e, n2) {
    var t = n2.defaultValue == null ? "" : n2.defaultValue, r = n2.checked != null ? n2.checked : n2.defaultChecked;
    t = tn(n2.value != null ? n2.value : t), e._wrapperState = { initialChecked: r, initialValue: t, controlled: n2.type === "checkbox" || n2.type === "radio" ? n2.checked != null : n2.value != null };
  }
  function So(e, n2) {
    n2 = n2.checked, n2 != null && vi(e, "checked", n2, false);
  }
  function Pl(e, n2) {
    So(e, n2);
    var t = tn(n2.value), r = n2.type;
    if (t != null) r === "number" ? (t === 0 && e.value === "" || e.value != t) && (e.value = "" + t) : e.value !== "" + t && (e.value = "" + t);
    else if (r === "submit" || r === "reset") {
      e.removeAttribute("value");
      return;
    }
    n2.hasOwnProperty("value") ? Ll(e, n2.type, t) : n2.hasOwnProperty("defaultValue") && Ll(e, n2.type, tn(n2.defaultValue)), n2.checked == null && n2.defaultChecked != null && (e.defaultChecked = !!n2.defaultChecked);
  }
  function fu(e, n2, t) {
    if (n2.hasOwnProperty("value") || n2.hasOwnProperty("defaultValue")) {
      var r = n2.type;
      if (!(r !== "submit" && r !== "reset" || n2.value !== void 0 && n2.value !== null)) return;
      n2 = "" + e._wrapperState.initialValue, t || n2 === e.value || (e.value = n2), e.defaultValue = n2;
    }
    t = e.name, t !== "" && (e.name = ""), e.defaultChecked = !!e._wrapperState.initialChecked, t !== "" && (e.name = t);
  }
  function Ll(e, n2, t) {
    (n2 !== "number" || mr(e.ownerDocument) !== e) && (t == null ? e.defaultValue = "" + e._wrapperState.initialValue : e.defaultValue !== "" + t && (e.defaultValue = "" + t));
  }
  var ut = Array.isArray;
  function In(e, n2, t, r) {
    if (e = e.options, n2) {
      n2 = {};
      for (var l3 = 0; l3 < t.length; l3++) n2["$" + t[l3]] = true;
      for (t = 0; t < e.length; t++) l3 = n2.hasOwnProperty("$" + e[t].value), e[t].selected !== l3 && (e[t].selected = l3), l3 && r && (e[t].defaultSelected = true);
    } else {
      for (t = "" + tn(t), n2 = null, l3 = 0; l3 < e.length; l3++) {
        if (e[l3].value === t) {
          e[l3].selected = true, r && (e[l3].defaultSelected = true);
          return;
        }
        n2 !== null || e[l3].disabled || (n2 = e[l3]);
      }
      n2 !== null && (n2.selected = true);
    }
  }
  function Tl(e, n2) {
    if (n2.dangerouslySetInnerHTML != null) throw Error(v2(91));
    return F3({}, n2, { value: void 0, defaultValue: void 0, children: "" + e._wrapperState.initialValue });
  }
  function du(e, n2) {
    var t = n2.value;
    if (t == null) {
      if (t = n2.children, n2 = n2.defaultValue, t != null) {
        if (n2 != null) throw Error(v2(92));
        if (ut(t)) {
          if (1 < t.length) throw Error(v2(93));
          t = t[0];
        }
        n2 = t;
      }
      n2 == null && (n2 = ""), t = n2;
    }
    e._wrapperState = { initialValue: tn(t) };
  }
  function ko(e, n2) {
    var t = tn(n2.value), r = tn(n2.defaultValue);
    t != null && (t = "" + t, t !== e.value && (e.value = t), n2.defaultValue == null && e.defaultValue !== t && (e.defaultValue = t)), r != null && (e.defaultValue = "" + r);
  }
  function pu(e) {
    var n2 = e.textContent;
    n2 === e._wrapperState.initialValue && n2 !== "" && n2 !== null && (e.value = n2);
  }
  function Eo(e) {
    switch (e) {
      case "svg":
        return "http://www.w3.org/2000/svg";
      case "math":
        return "http://www.w3.org/1998/Math/MathML";
      default:
        return "http://www.w3.org/1999/xhtml";
    }
  }
  function Ml(e, n2) {
    return e == null || e === "http://www.w3.org/1999/xhtml" ? Eo(n2) : e === "http://www.w3.org/2000/svg" && n2 === "foreignObject" ? "http://www.w3.org/1999/xhtml" : e;
  }
  var Bt, Co = function(e) {
    return typeof MSApp < "u" && MSApp.execUnsafeLocalFunction ? function(n2, t, r, l3) {
      MSApp.execUnsafeLocalFunction(function() {
        return e(n2, t, r, l3);
      });
    } : e;
  }(function(e, n2) {
    if (e.namespaceURI !== "http://www.w3.org/2000/svg" || "innerHTML" in e) e.innerHTML = n2;
    else {
      for (Bt = Bt || document.createElement("div"), Bt.innerHTML = "<svg>" + n2.valueOf().toString() + "</svg>", n2 = Bt.firstChild; e.firstChild; ) e.removeChild(e.firstChild);
      for (; n2.firstChild; ) e.appendChild(n2.firstChild);
    }
  });
  function wt(e, n2) {
    if (n2) {
      var t = e.firstChild;
      if (t && t === e.lastChild && t.nodeType === 3) {
        t.nodeValue = n2;
        return;
      }
    }
    e.textContent = n2;
  }
  var at = { animationIterationCount: true, aspectRatio: true, borderImageOutset: true, borderImageSlice: true, borderImageWidth: true, boxFlex: true, boxFlexGroup: true, boxOrdinalGroup: true, columnCount: true, columns: true, flex: true, flexGrow: true, flexPositive: true, flexShrink: true, flexNegative: true, flexOrder: true, gridArea: true, gridRow: true, gridRowEnd: true, gridRowSpan: true, gridRowStart: true, gridColumn: true, gridColumnEnd: true, gridColumnSpan: true, gridColumnStart: true, fontWeight: true, lineClamp: true, lineHeight: true, opacity: true, order: true, orphans: true, tabSize: true, widows: true, zIndex: true, zoom: true, fillOpacity: true, floodOpacity: true, stopOpacity: true, strokeDasharray: true, strokeDashoffset: true, strokeMiterlimit: true, strokeOpacity: true, strokeWidth: true }, Ma = ["Webkit", "ms", "Moz", "O"];
  Object.keys(at).forEach(function(e) {
    Ma.forEach(function(n2) {
      n2 = n2 + e.charAt(0).toUpperCase() + e.substring(1), at[n2] = at[e];
    });
  });
  function xo(e, n2, t) {
    return n2 == null || typeof n2 == "boolean" || n2 === "" ? "" : t || typeof n2 != "number" || n2 === 0 || at.hasOwnProperty(e) && at[e] ? ("" + n2).trim() : n2 + "px";
  }
  function No(e, n2) {
    e = e.style;
    for (var t in n2) if (n2.hasOwnProperty(t)) {
      var r = t.indexOf("--") === 0, l3 = xo(t, n2[t], r);
      t === "float" && (t = "cssFloat"), r ? e.setProperty(t, l3) : e[t] = l3;
    }
  }
  var Da = F3({ menuitem: true }, { area: true, base: true, br: true, col: true, embed: true, hr: true, img: true, input: true, keygen: true, link: true, meta: true, param: true, source: true, track: true, wbr: true });
  function Dl(e, n2) {
    if (n2) {
      if (Da[e] && (n2.children != null || n2.dangerouslySetInnerHTML != null)) throw Error(v2(137, e));
      if (n2.dangerouslySetInnerHTML != null) {
        if (n2.children != null) throw Error(v2(60));
        if (typeof n2.dangerouslySetInnerHTML != "object" || !("__html" in n2.dangerouslySetInnerHTML)) throw Error(v2(61));
      }
      if (n2.style != null && typeof n2.style != "object") throw Error(v2(62));
    }
  }
  function Ol(e, n2) {
    if (e.indexOf("-") === -1) return typeof n2.is == "string";
    switch (e) {
      case "annotation-xml":
      case "color-profile":
      case "font-face":
      case "font-face-src":
      case "font-face-uri":
      case "font-face-format":
      case "font-face-name":
      case "missing-glyph":
        return false;
      default:
        return true;
    }
  }
  var Rl = null;
  function Si(e) {
    return e = e.target || e.srcElement || globalThis, e.correspondingUseElement && (e = e.correspondingUseElement), e.nodeType === 3 ? e.parentNode : e;
  }
  var Fl = null, jn = null, Un = null;
  function mu(e) {
    if (e = jt(e)) {
      if (typeof Fl != "function") throw Error(v2(280));
      var n2 = e.stateNode;
      n2 && (n2 = Hr(n2), Fl(e.stateNode, e.type, n2));
    }
  }
  function _o(e) {
    jn ? Un ? Un.push(e) : Un = [e] : jn = e;
  }
  function zo() {
    if (jn) {
      var e = jn, n2 = Un;
      if (Un = jn = null, mu(e), n2) for (e = 0; e < n2.length; e++) mu(n2[e]);
    }
  }
  function Po(e, n2) {
    return e(n2);
  }
  function Lo() {
  }
  var rl = false;
  function To(e, n2, t) {
    if (rl) return e(n2, t);
    rl = true;
    try {
      return Po(e, n2, t);
    } finally {
      rl = false, (jn !== null || Un !== null) && (Lo(), zo());
    }
  }
  function St(e, n2) {
    var t = e.stateNode;
    if (t === null) return null;
    var r = Hr(t);
    if (r === null) return null;
    t = r[n2];
    e: switch (n2) {
      case "onClick":
      case "onClickCapture":
      case "onDoubleClick":
      case "onDoubleClickCapture":
      case "onMouseDown":
      case "onMouseDownCapture":
      case "onMouseMove":
      case "onMouseMoveCapture":
      case "onMouseUp":
      case "onMouseUpCapture":
      case "onMouseEnter":
        (r = !r.disabled) || (e = e.type, r = !(e === "button" || e === "input" || e === "select" || e === "textarea")), e = !r;
        break e;
      default:
        e = false;
    }
    if (e) return null;
    if (t && typeof t != "function") throw Error(v2(231, n2, typeof t));
    return t;
  }
  var Il = false;
  if (Fe2) try {
    En = {}, Object.defineProperty(En, "passive", { get: function() {
      Il = true;
    } }), globalThis.addEventListener("test", En, En), globalThis.removeEventListener("test", En, En);
  } catch {
    Il = false;
  }
  var En;
  function Oa(e, n2, t, r, l3, i2, u2, o, s) {
    var d3 = Array.prototype.slice.call(arguments, 3);
    try {
      n2.apply(t, d3);
    } catch (m2) {
      this.onError(m2);
    }
  }
  var ct = false, hr = null, vr = false, jl = null, Ra = { onError: function(e) {
    ct = true, hr = e;
  } };
  function Fa(e, n2, t, r, l3, i2, u2, o, s) {
    ct = false, hr = null, Oa.apply(Ra, arguments);
  }
  function Ia(e, n2, t, r, l3, i2, u2, o, s) {
    if (Fa.apply(this, arguments), ct) {
      if (ct) {
        var d3 = hr;
        ct = false, hr = null;
      } else throw Error(v2(198));
      vr || (vr = true, jl = d3);
    }
  }
  function kn(e) {
    var n2 = e, t = e;
    if (e.alternate) for (; n2.return; ) n2 = n2.return;
    else {
      e = n2;
      do
        n2 = e, (n2.flags & 4098) !== 0 && (t = n2.return), e = n2.return;
      while (e);
    }
    return n2.tag === 3 ? t : null;
  }
  function Mo(e) {
    if (e.tag === 13) {
      var n2 = e.memoizedState;
      if (n2 === null && (e = e.alternate, e !== null && (n2 = e.memoizedState)), n2 !== null) return n2.dehydrated;
    }
    return null;
  }
  function hu(e) {
    if (kn(e) !== e) throw Error(v2(188));
  }
  function ja(e) {
    var n2 = e.alternate;
    if (!n2) {
      if (n2 = kn(e), n2 === null) throw Error(v2(188));
      return n2 !== e ? null : e;
    }
    for (var t = e, r = n2; ; ) {
      var l3 = t.return;
      if (l3 === null) break;
      var i2 = l3.alternate;
      if (i2 === null) {
        if (r = l3.return, r !== null) {
          t = r;
          continue;
        }
        break;
      }
      if (l3.child === i2.child) {
        for (i2 = l3.child; i2; ) {
          if (i2 === t) return hu(l3), e;
          if (i2 === r) return hu(l3), n2;
          i2 = i2.sibling;
        }
        throw Error(v2(188));
      }
      if (t.return !== r.return) t = l3, r = i2;
      else {
        for (var u2 = false, o = l3.child; o; ) {
          if (o === t) {
            u2 = true, t = l3, r = i2;
            break;
          }
          if (o === r) {
            u2 = true, r = l3, t = i2;
            break;
          }
          o = o.sibling;
        }
        if (!u2) {
          for (o = i2.child; o; ) {
            if (o === t) {
              u2 = true, t = i2, r = l3;
              break;
            }
            if (o === r) {
              u2 = true, r = i2, t = l3;
              break;
            }
            o = o.sibling;
          }
          if (!u2) throw Error(v2(189));
        }
      }
      if (t.alternate !== r) throw Error(v2(190));
    }
    if (t.tag !== 3) throw Error(v2(188));
    return t.stateNode.current === t ? e : n2;
  }
  function Do(e) {
    return e = ja(e), e !== null ? Oo(e) : null;
  }
  function Oo(e) {
    if (e.tag === 5 || e.tag === 6) return e;
    for (e = e.child; e !== null; ) {
      var n2 = Oo(e);
      if (n2 !== null) return n2;
      e = e.sibling;
    }
    return null;
  }
  var Ro = ae3.unstable_scheduleCallback, vu = ae3.unstable_cancelCallback, Ua = ae3.unstable_shouldYield, Va = ae3.unstable_requestPaint, U3 = ae3.unstable_now, Aa = ae3.unstable_getCurrentPriorityLevel, ki = ae3.unstable_ImmediatePriority, Fo = ae3.unstable_UserBlockingPriority, yr = ae3.unstable_NormalPriority, Ba = ae3.unstable_LowPriority, Io = ae3.unstable_IdlePriority, Ur = null, Pe3 = null;
  function Ha(e) {
    if (Pe3 && typeof Pe3.onCommitFiberRoot == "function") try {
      Pe3.onCommitFiberRoot(Ur, e, void 0, (e.current.flags & 128) === 128);
    } catch {
    }
  }
  var Ee2 = Math.clz32 ? Math.clz32 : $a, Wa = Math.log, Qa = Math.LN2;
  function $a(e) {
    return e >>>= 0, e === 0 ? 32 : 31 - (Wa(e) / Qa | 0) | 0;
  }
  var Ht = 64, Wt = 4194304;
  function ot(e) {
    switch (e & -e) {
      case 1:
        return 1;
      case 2:
        return 2;
      case 4:
        return 4;
      case 8:
        return 8;
      case 16:
        return 16;
      case 32:
        return 32;
      case 64:
      case 128:
      case 256:
      case 512:
      case 1024:
      case 2048:
      case 4096:
      case 8192:
      case 16384:
      case 32768:
      case 65536:
      case 131072:
      case 262144:
      case 524288:
      case 1048576:
      case 2097152:
        return e & 4194240;
      case 4194304:
      case 8388608:
      case 16777216:
      case 33554432:
      case 67108864:
        return e & 130023424;
      case 134217728:
        return 134217728;
      case 268435456:
        return 268435456;
      case 536870912:
        return 536870912;
      case 1073741824:
        return 1073741824;
      default:
        return e;
    }
  }
  function gr(e, n2) {
    var t = e.pendingLanes;
    if (t === 0) return 0;
    var r = 0, l3 = e.suspendedLanes, i2 = e.pingedLanes, u2 = t & 268435455;
    if (u2 !== 0) {
      var o = u2 & ~l3;
      o !== 0 ? r = ot(o) : (i2 &= u2, i2 !== 0 && (r = ot(i2)));
    } else u2 = t & ~l3, u2 !== 0 ? r = ot(u2) : i2 !== 0 && (r = ot(i2));
    if (r === 0) return 0;
    if (n2 !== 0 && n2 !== r && (n2 & l3) === 0 && (l3 = r & -r, i2 = n2 & -n2, l3 >= i2 || l3 === 16 && (i2 & 4194240) !== 0)) return n2;
    if ((r & 4) !== 0 && (r |= t & 16), n2 = e.entangledLanes, n2 !== 0) for (e = e.entanglements, n2 &= r; 0 < n2; ) t = 31 - Ee2(n2), l3 = 1 << t, r |= e[t], n2 &= ~l3;
    return r;
  }
  function Ka(e, n2) {
    switch (e) {
      case 1:
      case 2:
      case 4:
        return n2 + 250;
      case 8:
      case 16:
      case 32:
      case 64:
      case 128:
      case 256:
      case 512:
      case 1024:
      case 2048:
      case 4096:
      case 8192:
      case 16384:
      case 32768:
      case 65536:
      case 131072:
      case 262144:
      case 524288:
      case 1048576:
      case 2097152:
        return n2 + 5e3;
      case 4194304:
      case 8388608:
      case 16777216:
      case 33554432:
      case 67108864:
        return -1;
      case 134217728:
      case 268435456:
      case 536870912:
      case 1073741824:
        return -1;
      default:
        return -1;
    }
  }
  function Ya(e, n2) {
    for (var t = e.suspendedLanes, r = e.pingedLanes, l3 = e.expirationTimes, i2 = e.pendingLanes; 0 < i2; ) {
      var u2 = 31 - Ee2(i2), o = 1 << u2, s = l3[u2];
      s === -1 ? ((o & t) === 0 || (o & r) !== 0) && (l3[u2] = Ka(o, n2)) : s <= n2 && (e.expiredLanes |= o), i2 &= ~o;
    }
  }
  function Ul(e) {
    return e = e.pendingLanes & -1073741825, e !== 0 ? e : e & 1073741824 ? 1073741824 : 0;
  }
  function jo() {
    var e = Ht;
    return Ht <<= 1, (Ht & 4194240) === 0 && (Ht = 64), e;
  }
  function ll(e) {
    for (var n2 = [], t = 0; 31 > t; t++) n2.push(e);
    return n2;
  }
  function Ft(e, n2, t) {
    e.pendingLanes |= n2, n2 !== 536870912 && (e.suspendedLanes = 0, e.pingedLanes = 0), e = e.eventTimes, n2 = 31 - Ee2(n2), e[n2] = t;
  }
  function Xa(e, n2) {
    var t = e.pendingLanes & ~n2;
    e.pendingLanes = n2, e.suspendedLanes = 0, e.pingedLanes = 0, e.expiredLanes &= n2, e.mutableReadLanes &= n2, e.entangledLanes &= n2, n2 = e.entanglements;
    var r = e.eventTimes;
    for (e = e.expirationTimes; 0 < t; ) {
      var l3 = 31 - Ee2(t), i2 = 1 << l3;
      n2[l3] = 0, r[l3] = -1, e[l3] = -1, t &= ~i2;
    }
  }
  function Ei(e, n2) {
    var t = e.entangledLanes |= n2;
    for (e = e.entanglements; t; ) {
      var r = 31 - Ee2(t), l3 = 1 << r;
      l3 & n2 | e[r] & n2 && (e[r] |= n2), t &= ~l3;
    }
  }
  var P = 0;
  function Uo(e) {
    return e &= -e, 1 < e ? 4 < e ? (e & 268435455) !== 0 ? 16 : 536870912 : 4 : 1;
  }
  var Vo, Ci, Ao, Bo, Ho, Vl = false, Qt = [], Xe = null, Ge = null, Ze = null, kt = /* @__PURE__ */ new Map(), Et = /* @__PURE__ */ new Map(), Qe = [], Ga = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");
  function yu(e, n2) {
    switch (e) {
      case "focusin":
      case "focusout":
        Xe = null;
        break;
      case "dragenter":
      case "dragleave":
        Ge = null;
        break;
      case "mouseover":
      case "mouseout":
        Ze = null;
        break;
      case "pointerover":
      case "pointerout":
        kt.delete(n2.pointerId);
        break;
      case "gotpointercapture":
      case "lostpointercapture":
        Et.delete(n2.pointerId);
    }
  }
  function qn(e, n2, t, r, l3, i2) {
    return e === null || e.nativeEvent !== i2 ? (e = { blockedOn: n2, domEventName: t, eventSystemFlags: r, nativeEvent: i2, targetContainers: [l3] }, n2 !== null && (n2 = jt(n2), n2 !== null && Ci(n2)), e) : (e.eventSystemFlags |= r, n2 = e.targetContainers, l3 !== null && n2.indexOf(l3) === -1 && n2.push(l3), e);
  }
  function Za(e, n2, t, r, l3) {
    switch (n2) {
      case "focusin":
        return Xe = qn(Xe, e, n2, t, r, l3), true;
      case "dragenter":
        return Ge = qn(Ge, e, n2, t, r, l3), true;
      case "mouseover":
        return Ze = qn(Ze, e, n2, t, r, l3), true;
      case "pointerover":
        var i2 = l3.pointerId;
        return kt.set(i2, qn(kt.get(i2) || null, e, n2, t, r, l3)), true;
      case "gotpointercapture":
        return i2 = l3.pointerId, Et.set(i2, qn(Et.get(i2) || null, e, n2, t, r, l3)), true;
    }
    return false;
  }
  function Wo(e) {
    var n2 = cn(e.target);
    if (n2 !== null) {
      var t = kn(n2);
      if (t !== null) {
        if (n2 = t.tag, n2 === 13) {
          if (n2 = Mo(t), n2 !== null) {
            e.blockedOn = n2, Ho(e.priority, function() {
              Ao(t);
            });
            return;
          }
        } else if (n2 === 3 && t.stateNode.current.memoizedState.isDehydrated) {
          e.blockedOn = t.tag === 3 ? t.stateNode.containerInfo : null;
          return;
        }
      }
    }
    e.blockedOn = null;
  }
  function lr(e) {
    if (e.blockedOn !== null) return false;
    for (var n2 = e.targetContainers; 0 < n2.length; ) {
      var t = Al(e.domEventName, e.eventSystemFlags, n2[0], e.nativeEvent);
      if (t === null) {
        t = e.nativeEvent;
        var r = new t.constructor(t.type, t);
        Rl = r, t.target.dispatchEvent(r), Rl = null;
      } else return n2 = jt(t), n2 !== null && Ci(n2), e.blockedOn = t, false;
      n2.shift();
    }
    return true;
  }
  function gu(e, n2, t) {
    lr(e) && t.delete(n2);
  }
  function Ja() {
    Vl = false, Xe !== null && lr(Xe) && (Xe = null), Ge !== null && lr(Ge) && (Ge = null), Ze !== null && lr(Ze) && (Ze = null), kt.forEach(gu), Et.forEach(gu);
  }
  function bn(e, n2) {
    e.blockedOn === n2 && (e.blockedOn = null, Vl || (Vl = true, ae3.unstable_scheduleCallback(ae3.unstable_NormalPriority, Ja)));
  }
  function Ct(e) {
    function n2(l3) {
      return bn(l3, e);
    }
    if (0 < Qt.length) {
      bn(Qt[0], e);
      for (var t = 1; t < Qt.length; t++) {
        var r = Qt[t];
        r.blockedOn === e && (r.blockedOn = null);
      }
    }
    for (Xe !== null && bn(Xe, e), Ge !== null && bn(Ge, e), Ze !== null && bn(Ze, e), kt.forEach(n2), Et.forEach(n2), t = 0; t < Qe.length; t++) r = Qe[t], r.blockedOn === e && (r.blockedOn = null);
    for (; 0 < Qe.length && (t = Qe[0], t.blockedOn === null); ) Wo(t), t.blockedOn === null && Qe.shift();
  }
  var Vn = Ve2.ReactCurrentBatchConfig, wr = true;
  function qa(e, n2, t, r) {
    var l3 = P, i2 = Vn.transition;
    Vn.transition = null;
    try {
      P = 1, xi(e, n2, t, r);
    } finally {
      P = l3, Vn.transition = i2;
    }
  }
  function ba(e, n2, t, r) {
    var l3 = P, i2 = Vn.transition;
    Vn.transition = null;
    try {
      P = 4, xi(e, n2, t, r);
    } finally {
      P = l3, Vn.transition = i2;
    }
  }
  function xi(e, n2, t, r) {
    if (wr) {
      var l3 = Al(e, n2, t, r);
      if (l3 === null) fl(e, n2, r, Sr, t), yu(e, r);
      else if (Za(l3, e, n2, t, r)) r.stopPropagation();
      else if (yu(e, r), n2 & 4 && -1 < Ga.indexOf(e)) {
        for (; l3 !== null; ) {
          var i2 = jt(l3);
          if (i2 !== null && Vo(i2), i2 = Al(e, n2, t, r), i2 === null && fl(e, n2, r, Sr, t), i2 === l3) break;
          l3 = i2;
        }
        l3 !== null && r.stopPropagation();
      } else fl(e, n2, r, null, t);
    }
  }
  var Sr = null;
  function Al(e, n2, t, r) {
    if (Sr = null, e = Si(r), e = cn(e), e !== null) if (n2 = kn(e), n2 === null) e = null;
    else if (t = n2.tag, t === 13) {
      if (e = Mo(n2), e !== null) return e;
      e = null;
    } else if (t === 3) {
      if (n2.stateNode.current.memoizedState.isDehydrated) return n2.tag === 3 ? n2.stateNode.containerInfo : null;
      e = null;
    } else n2 !== e && (e = null);
    return Sr = e, null;
  }
  function Qo(e) {
    switch (e) {
      case "cancel":
      case "click":
      case "close":
      case "contextmenu":
      case "copy":
      case "cut":
      case "auxclick":
      case "dblclick":
      case "dragend":
      case "dragstart":
      case "drop":
      case "focusin":
      case "focusout":
      case "input":
      case "invalid":
      case "keydown":
      case "keypress":
      case "keyup":
      case "mousedown":
      case "mouseup":
      case "paste":
      case "pause":
      case "play":
      case "pointercancel":
      case "pointerdown":
      case "pointerup":
      case "ratechange":
      case "reset":
      case "resize":
      case "seeked":
      case "submit":
      case "touchcancel":
      case "touchend":
      case "touchstart":
      case "volumechange":
      case "change":
      case "selectionchange":
      case "textInput":
      case "compositionstart":
      case "compositionend":
      case "compositionupdate":
      case "beforeblur":
      case "afterblur":
      case "beforeinput":
      case "blur":
      case "fullscreenchange":
      case "focus":
      case "hashchange":
      case "popstate":
      case "select":
      case "selectstart":
        return 1;
      case "drag":
      case "dragenter":
      case "dragexit":
      case "dragleave":
      case "dragover":
      case "mousemove":
      case "mouseout":
      case "mouseover":
      case "pointermove":
      case "pointerout":
      case "pointerover":
      case "scroll":
      case "toggle":
      case "touchmove":
      case "wheel":
      case "mouseenter":
      case "mouseleave":
      case "pointerenter":
      case "pointerleave":
        return 4;
      case "message":
        switch (Aa()) {
          case ki:
            return 1;
          case Fo:
            return 4;
          case yr:
          case Ba:
            return 16;
          case Io:
            return 536870912;
          default:
            return 16;
        }
      default:
        return 16;
    }
  }
  var Ke = null, Ni = null, ir = null;
  function $o() {
    if (ir) return ir;
    var e, n2 = Ni, t = n2.length, r, l3 = "value" in Ke ? Ke.value : Ke.textContent, i2 = l3.length;
    for (e = 0; e < t && n2[e] === l3[e]; e++) ;
    var u2 = t - e;
    for (r = 1; r <= u2 && n2[t - r] === l3[i2 - r]; r++) ;
    return ir = l3.slice(e, 1 < r ? 1 - r : void 0);
  }
  function ur(e) {
    var n2 = e.keyCode;
    return "charCode" in e ? (e = e.charCode, e === 0 && n2 === 13 && (e = 13)) : e = n2, e === 10 && (e = 13), 32 <= e || e === 13 ? e : 0;
  }
  function $t() {
    return true;
  }
  function wu() {
    return false;
  }
  function ce2(e) {
    function n2(t, r, l3, i2, u2) {
      this._reactName = t, this._targetInst = l3, this.type = r, this.nativeEvent = i2, this.target = u2, this.currentTarget = null;
      for (var o in e) e.hasOwnProperty(o) && (t = e[o], this[o] = t ? t(i2) : i2[o]);
      return this.isDefaultPrevented = (i2.defaultPrevented != null ? i2.defaultPrevented : i2.returnValue === false) ? $t : wu, this.isPropagationStopped = wu, this;
    }
    return F3(n2.prototype, { preventDefault: function() {
      this.defaultPrevented = true;
      var t = this.nativeEvent;
      t && (t.preventDefault ? t.preventDefault() : typeof t.returnValue != "unknown" && (t.returnValue = false), this.isDefaultPrevented = $t);
    }, stopPropagation: function() {
      var t = this.nativeEvent;
      t && (t.stopPropagation ? t.stopPropagation() : typeof t.cancelBubble != "unknown" && (t.cancelBubble = true), this.isPropagationStopped = $t);
    }, persist: function() {
    }, isPersistent: $t }), n2;
  }
  var Gn = { eventPhase: 0, bubbles: 0, cancelable: 0, timeStamp: function(e) {
    return e.timeStamp || Date.now();
  }, defaultPrevented: 0, isTrusted: 0 }, _i = ce2(Gn), It = F3({}, Gn, { view: 0, detail: 0 }), ec = ce2(It), il, ul, et, Vr = F3({}, It, { screenX: 0, screenY: 0, clientX: 0, clientY: 0, pageX: 0, pageY: 0, ctrlKey: 0, shiftKey: 0, altKey: 0, metaKey: 0, getModifierState: zi, button: 0, buttons: 0, relatedTarget: function(e) {
    return e.relatedTarget === void 0 ? e.fromElement === e.srcElement ? e.toElement : e.fromElement : e.relatedTarget;
  }, movementX: function(e) {
    return "movementX" in e ? e.movementX : (e !== et && (et && e.type === "mousemove" ? (il = e.screenX - et.screenX, ul = e.screenY - et.screenY) : ul = il = 0, et = e), il);
  }, movementY: function(e) {
    return "movementY" in e ? e.movementY : ul;
  } }), Su = ce2(Vr), nc = F3({}, Vr, { dataTransfer: 0 }), tc = ce2(nc), rc = F3({}, It, { relatedTarget: 0 }), ol = ce2(rc), lc = F3({}, Gn, { animationName: 0, elapsedTime: 0, pseudoElement: 0 }), ic = ce2(lc), uc = F3({}, Gn, { clipboardData: function(e) {
    return "clipboardData" in e ? e.clipboardData : globalThis.clipboardData;
  } }), oc = ce2(uc), sc = F3({}, Gn, { data: 0 }), ku = ce2(sc), ac = { Esc: "Escape", Spacebar: " ", Left: "ArrowLeft", Up: "ArrowUp", Right: "ArrowRight", Down: "ArrowDown", Del: "Delete", Win: "OS", Menu: "ContextMenu", Apps: "ContextMenu", Scroll: "ScrollLock", MozPrintableKey: "Unidentified" }, cc = { 8: "Backspace", 9: "Tab", 12: "Clear", 13: "Enter", 16: "Shift", 17: "Control", 18: "Alt", 19: "Pause", 20: "CapsLock", 27: "Escape", 32: " ", 33: "PageUp", 34: "PageDown", 35: "End", 36: "Home", 37: "ArrowLeft", 38: "ArrowUp", 39: "ArrowRight", 40: "ArrowDown", 45: "Insert", 46: "Delete", 112: "F1", 113: "F2", 114: "F3", 115: "F4", 116: "F5", 117: "F6", 118: "F7", 119: "F8", 120: "F9", 121: "F10", 122: "F11", 123: "F12", 144: "NumLock", 145: "ScrollLock", 224: "Meta" }, fc = { Alt: "altKey", Control: "ctrlKey", Meta: "metaKey", Shift: "shiftKey" };
  function dc(e) {
    var n2 = this.nativeEvent;
    return n2.getModifierState ? n2.getModifierState(e) : (e = fc[e]) ? !!n2[e] : false;
  }
  function zi() {
    return dc;
  }
  var pc = F3({}, It, { key: function(e) {
    if (e.key) {
      var n2 = ac[e.key] || e.key;
      if (n2 !== "Unidentified") return n2;
    }
    return e.type === "keypress" ? (e = ur(e), e === 13 ? "Enter" : String.fromCharCode(e)) : e.type === "keydown" || e.type === "keyup" ? cc[e.keyCode] || "Unidentified" : "";
  }, code: 0, location: 0, ctrlKey: 0, shiftKey: 0, altKey: 0, metaKey: 0, repeat: 0, locale: 0, getModifierState: zi, charCode: function(e) {
    return e.type === "keypress" ? ur(e) : 0;
  }, keyCode: function(e) {
    return e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
  }, which: function(e) {
    return e.type === "keypress" ? ur(e) : e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
  } }), mc = ce2(pc), hc = F3({}, Vr, { pointerId: 0, width: 0, height: 0, pressure: 0, tangentialPressure: 0, tiltX: 0, tiltY: 0, twist: 0, pointerType: 0, isPrimary: 0 }), Eu = ce2(hc), vc = F3({}, It, { touches: 0, targetTouches: 0, changedTouches: 0, altKey: 0, metaKey: 0, ctrlKey: 0, shiftKey: 0, getModifierState: zi }), yc = ce2(vc), gc = F3({}, Gn, { propertyName: 0, elapsedTime: 0, pseudoElement: 0 }), wc = ce2(gc), Sc = F3({}, Vr, { deltaX: function(e) {
    return "deltaX" in e ? e.deltaX : "wheelDeltaX" in e ? -e.wheelDeltaX : 0;
  }, deltaY: function(e) {
    return "deltaY" in e ? e.deltaY : "wheelDeltaY" in e ? -e.wheelDeltaY : "wheelDelta" in e ? -e.wheelDelta : 0;
  }, deltaZ: 0, deltaMode: 0 }), kc = ce2(Sc), Ec = [9, 13, 27, 32], Pi = Fe2 && "CompositionEvent" in globalThis, ft = null;
  Fe2 && "documentMode" in document && (ft = document.documentMode);
  var Cc = Fe2 && "TextEvent" in globalThis && !ft, Ko = Fe2 && (!Pi || ft && 8 < ft && 11 >= ft), Cu = " ", xu = false;
  function Yo(e, n2) {
    switch (e) {
      case "keyup":
        return Ec.indexOf(n2.keyCode) !== -1;
      case "keydown":
        return n2.keyCode !== 229;
      case "keypress":
      case "mousedown":
      case "focusout":
        return true;
      default:
        return false;
    }
  }
  function Xo(e) {
    return e = e.detail, typeof e == "object" && "data" in e ? e.data : null;
  }
  var _n = false;
  function xc(e, n2) {
    switch (e) {
      case "compositionend":
        return Xo(n2);
      case "keypress":
        return n2.which !== 32 ? null : (xu = true, Cu);
      case "textInput":
        return e = n2.data, e === Cu && xu ? null : e;
      default:
        return null;
    }
  }
  function Nc(e, n2) {
    if (_n) return e === "compositionend" || !Pi && Yo(e, n2) ? (e = $o(), ir = Ni = Ke = null, _n = false, e) : null;
    switch (e) {
      case "paste":
        return null;
      case "keypress":
        if (!(n2.ctrlKey || n2.altKey || n2.metaKey) || n2.ctrlKey && n2.altKey) {
          if (n2.char && 1 < n2.char.length) return n2.char;
          if (n2.which) return String.fromCharCode(n2.which);
        }
        return null;
      case "compositionend":
        return Ko && n2.locale !== "ko" ? null : n2.data;
      default:
        return null;
    }
  }
  var _c = { color: true, date: true, datetime: true, "datetime-local": true, email: true, month: true, number: true, password: true, range: true, search: true, tel: true, text: true, time: true, url: true, week: true };
  function Nu(e) {
    var n2 = e && e.nodeName && e.nodeName.toLowerCase();
    return n2 === "input" ? !!_c[e.type] : n2 === "textarea";
  }
  function Go(e, n2, t, r) {
    _o(r), n2 = kr(n2, "onChange"), 0 < n2.length && (t = new _i("onChange", "change", null, t, r), e.push({ event: t, listeners: n2 }));
  }
  var dt = null, xt = null;
  function zc(e) {
    us(e, 0);
  }
  function Ar(e) {
    var n2 = Ln(e);
    if (wo(n2)) return e;
  }
  function Pc(e, n2) {
    if (e === "change") return n2;
  }
  var Zo = false;
  Fe2 && (Fe2 ? (Yt = "oninput" in document, Yt || (sl = document.createElement("div"), sl.setAttribute("oninput", "return;"), Yt = typeof sl.oninput == "function"), Kt = Yt) : Kt = false, Zo = Kt && (!document.documentMode || 9 < document.documentMode));
  var Kt, Yt, sl;
  function _u() {
    dt && (dt.detachEvent("onpropertychange", Jo), xt = dt = null);
  }
  function Jo(e) {
    if (e.propertyName === "value" && Ar(xt)) {
      var n2 = [];
      Go(n2, xt, e, Si(e)), To(zc, n2);
    }
  }
  function Lc(e, n2, t) {
    e === "focusin" ? (_u(), dt = n2, xt = t, dt.attachEvent("onpropertychange", Jo)) : e === "focusout" && _u();
  }
  function Tc(e) {
    if (e === "selectionchange" || e === "keyup" || e === "keydown") return Ar(xt);
  }
  function Mc(e, n2) {
    if (e === "click") return Ar(n2);
  }
  function Dc(e, n2) {
    if (e === "input" || e === "change") return Ar(n2);
  }
  function Oc(e, n2) {
    return e === n2 && (e !== 0 || 1 / e === 1 / n2) || e !== e && n2 !== n2;
  }
  var xe3 = typeof Object.is == "function" ? Object.is : Oc;
  function Nt(e, n2) {
    if (xe3(e, n2)) return true;
    if (typeof e != "object" || e === null || typeof n2 != "object" || n2 === null) return false;
    var t = Object.keys(e), r = Object.keys(n2);
    if (t.length !== r.length) return false;
    for (r = 0; r < t.length; r++) {
      var l3 = t[r];
      if (!El.call(n2, l3) || !xe3(e[l3], n2[l3])) return false;
    }
    return true;
  }
  function zu(e) {
    for (; e && e.firstChild; ) e = e.firstChild;
    return e;
  }
  function Pu(e, n2) {
    var t = zu(e);
    e = 0;
    for (var r; t; ) {
      if (t.nodeType === 3) {
        if (r = e + t.textContent.length, e <= n2 && r >= n2) return { node: t, offset: n2 - e };
        e = r;
      }
      e: {
        for (; t; ) {
          if (t.nextSibling) {
            t = t.nextSibling;
            break e;
          }
          t = t.parentNode;
        }
        t = void 0;
      }
      t = zu(t);
    }
  }
  function qo(e, n2) {
    return e && n2 ? e === n2 ? true : e && e.nodeType === 3 ? false : n2 && n2.nodeType === 3 ? qo(e, n2.parentNode) : "contains" in e ? e.contains(n2) : e.compareDocumentPosition ? !!(e.compareDocumentPosition(n2) & 16) : false : false;
  }
  function bo() {
    for (var e = globalThis, n2 = mr(); n2 instanceof e.HTMLIFrameElement; ) {
      try {
        var t = typeof n2.contentWindow.location.href == "string";
      } catch {
        t = false;
      }
      if (t) e = n2.contentWindow;
      else break;
      n2 = mr(e.document);
    }
    return n2;
  }
  function Li(e) {
    var n2 = e && e.nodeName && e.nodeName.toLowerCase();
    return n2 && (n2 === "input" && (e.type === "text" || e.type === "search" || e.type === "tel" || e.type === "url" || e.type === "password") || n2 === "textarea" || e.contentEditable === "true");
  }
  function Rc(e) {
    var n2 = bo(), t = e.focusedElem, r = e.selectionRange;
    if (n2 !== t && t && t.ownerDocument && qo(t.ownerDocument.documentElement, t)) {
      if (r !== null && Li(t)) {
        if (n2 = r.start, e = r.end, e === void 0 && (e = n2), "selectionStart" in t) t.selectionStart = n2, t.selectionEnd = Math.min(e, t.value.length);
        else if (e = (n2 = t.ownerDocument || document) && n2.defaultView || globalThis, e.getSelection) {
          e = e.getSelection();
          var l3 = t.textContent.length, i2 = Math.min(r.start, l3);
          r = r.end === void 0 ? i2 : Math.min(r.end, l3), !e.extend && i2 > r && (l3 = r, r = i2, i2 = l3), l3 = Pu(t, i2);
          var u2 = Pu(t, r);
          l3 && u2 && (e.rangeCount !== 1 || e.anchorNode !== l3.node || e.anchorOffset !== l3.offset || e.focusNode !== u2.node || e.focusOffset !== u2.offset) && (n2 = n2.createRange(), n2.setStart(l3.node, l3.offset), e.removeAllRanges(), i2 > r ? (e.addRange(n2), e.extend(u2.node, u2.offset)) : (n2.setEnd(u2.node, u2.offset), e.addRange(n2)));
        }
      }
      for (n2 = [], e = t; e = e.parentNode; ) e.nodeType === 1 && n2.push({ element: e, left: e.scrollLeft, top: e.scrollTop });
      for (typeof t.focus == "function" && t.focus(), t = 0; t < n2.length; t++) e = n2[t], e.element.scrollLeft = e.left, e.element.scrollTop = e.top;
    }
  }
  var Fc = Fe2 && "documentMode" in document && 11 >= document.documentMode, zn = null, Bl = null, pt = null, Hl = false;
  function Lu(e, n2, t) {
    var r = t.window === t ? t.document : t.nodeType === 9 ? t : t.ownerDocument;
    Hl || zn == null || zn !== mr(r) || (r = zn, "selectionStart" in r && Li(r) ? r = { start: r.selectionStart, end: r.selectionEnd } : (r = (r.ownerDocument && r.ownerDocument.defaultView || globalThis).getSelection(), r = { anchorNode: r.anchorNode, anchorOffset: r.anchorOffset, focusNode: r.focusNode, focusOffset: r.focusOffset }), pt && Nt(pt, r) || (pt = r, r = kr(Bl, "onSelect"), 0 < r.length && (n2 = new _i("onSelect", "select", null, n2, t), e.push({ event: n2, listeners: r }), n2.target = zn)));
  }
  function Xt(e, n2) {
    var t = {};
    return t[e.toLowerCase()] = n2.toLowerCase(), t["Webkit" + e] = "webkit" + n2, t["Moz" + e] = "moz" + n2, t;
  }
  var Pn = { animationend: Xt("Animation", "AnimationEnd"), animationiteration: Xt("Animation", "AnimationIteration"), animationstart: Xt("Animation", "AnimationStart"), transitionend: Xt("Transition", "TransitionEnd") }, al = {}, es = {};
  Fe2 && (es = document.createElement("div").style, "AnimationEvent" in globalThis || (delete Pn.animationend.animation, delete Pn.animationiteration.animation, delete Pn.animationstart.animation), "TransitionEvent" in globalThis || delete Pn.transitionend.transition);
  function Br(e) {
    if (al[e]) return al[e];
    if (!Pn[e]) return e;
    var n2 = Pn[e], t;
    for (t in n2) if (n2.hasOwnProperty(t) && t in es) return al[e] = n2[t];
    return e;
  }
  var ns = Br("animationend"), ts = Br("animationiteration"), rs = Br("animationstart"), ls = Br("transitionend"), is = /* @__PURE__ */ new Map(), Tu = "abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");
  function ln(e, n2) {
    is.set(e, n2), Sn(n2, [e]);
  }
  for (Gt = 0; Gt < Tu.length; Gt++) Zt = Tu[Gt], Mu = Zt.toLowerCase(), Du = Zt[0].toUpperCase() + Zt.slice(1), ln(Mu, "on" + Du);
  var Zt, Mu, Du, Gt;
  ln(ns, "onAnimationEnd");
  ln(ts, "onAnimationIteration");
  ln(rs, "onAnimationStart");
  ln("dblclick", "onDoubleClick");
  ln("focusin", "onFocus");
  ln("focusout", "onBlur");
  ln(ls, "onTransitionEnd");
  Hn("onMouseEnter", ["mouseout", "mouseover"]);
  Hn("onMouseLeave", ["mouseout", "mouseover"]);
  Hn("onPointerEnter", ["pointerout", "pointerover"]);
  Hn("onPointerLeave", ["pointerout", "pointerover"]);
  Sn("onChange", "change click focusin focusout input keydown keyup selectionchange".split(" "));
  Sn("onSelect", "focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));
  Sn("onBeforeInput", ["compositionend", "keypress", "textInput", "paste"]);
  Sn("onCompositionEnd", "compositionend focusout keydown keypress keyup mousedown".split(" "));
  Sn("onCompositionStart", "compositionstart focusout keydown keypress keyup mousedown".split(" "));
  Sn("onCompositionUpdate", "compositionupdate focusout keydown keypress keyup mousedown".split(" "));
  var st = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "), Ic = new Set("cancel close invalid load scroll toggle".split(" ").concat(st));
  function Ou(e, n2, t) {
    var r = e.type || "unknown-event";
    e.currentTarget = t, Ia(r, n2, void 0, e), e.currentTarget = null;
  }
  function us(e, n2) {
    n2 = (n2 & 4) !== 0;
    for (var t = 0; t < e.length; t++) {
      var r = e[t], l3 = r.event;
      r = r.listeners;
      e: {
        var i2 = void 0;
        if (n2) for (var u2 = r.length - 1; 0 <= u2; u2--) {
          var o = r[u2], s = o.instance, d3 = o.currentTarget;
          if (o = o.listener, s !== i2 && l3.isPropagationStopped()) break e;
          Ou(l3, o, d3), i2 = s;
        }
        else for (u2 = 0; u2 < r.length; u2++) {
          if (o = r[u2], s = o.instance, d3 = o.currentTarget, o = o.listener, s !== i2 && l3.isPropagationStopped()) break e;
          Ou(l3, o, d3), i2 = s;
        }
      }
    }
    if (vr) throw e = jl, vr = false, jl = null, e;
  }
  function T2(e, n2) {
    var t = n2[Yl];
    t === void 0 && (t = n2[Yl] = /* @__PURE__ */ new Set());
    var r = e + "__bubble";
    t.has(r) || (os(n2, e, 2, false), t.add(r));
  }
  function cl(e, n2, t) {
    var r = 0;
    n2 && (r |= 4), os(t, e, r, n2);
  }
  var Jt = "_reactListening" + Math.random().toString(36).slice(2);
  function _t(e) {
    if (!e[Jt]) {
      e[Jt] = true, mo.forEach(function(t) {
        t !== "selectionchange" && (Ic.has(t) || cl(t, false, e), cl(t, true, e));
      });
      var n2 = e.nodeType === 9 ? e : e.ownerDocument;
      n2 === null || n2[Jt] || (n2[Jt] = true, cl("selectionchange", false, n2));
    }
  }
  function os(e, n2, t, r) {
    switch (Qo(n2)) {
      case 1:
        var l3 = qa;
        break;
      case 4:
        l3 = ba;
        break;
      default:
        l3 = xi;
    }
    t = l3.bind(null, n2, t, e), l3 = void 0, !Il || n2 !== "touchstart" && n2 !== "touchmove" && n2 !== "wheel" || (l3 = true), r ? l3 !== void 0 ? e.addEventListener(n2, t, { capture: true, passive: l3 }) : e.addEventListener(n2, t, true) : l3 !== void 0 ? e.addEventListener(n2, t, { passive: l3 }) : e.addEventListener(n2, t, false);
  }
  function fl(e, n2, t, r, l3) {
    var i2 = r;
    if ((n2 & 1) === 0 && (n2 & 2) === 0 && r !== null) e: for (; ; ) {
      if (r === null) return;
      var u2 = r.tag;
      if (u2 === 3 || u2 === 4) {
        var o = r.stateNode.containerInfo;
        if (o === l3 || o.nodeType === 8 && o.parentNode === l3) break;
        if (u2 === 4) for (u2 = r.return; u2 !== null; ) {
          var s = u2.tag;
          if ((s === 3 || s === 4) && (s = u2.stateNode.containerInfo, s === l3 || s.nodeType === 8 && s.parentNode === l3)) return;
          u2 = u2.return;
        }
        for (; o !== null; ) {
          if (u2 = cn(o), u2 === null) return;
          if (s = u2.tag, s === 5 || s === 6) {
            r = i2 = u2;
            continue e;
          }
          o = o.parentNode;
        }
      }
      r = r.return;
    }
    To(function() {
      var d3 = i2, m2 = Si(t), h3 = [];
      e: {
        var p = is.get(e);
        if (p !== void 0) {
          var g2 = _i, S2 = e;
          switch (e) {
            case "keypress":
              if (ur(t) === 0) break e;
            case "keydown":
            case "keyup":
              g2 = mc;
              break;
            case "focusin":
              S2 = "focus", g2 = ol;
              break;
            case "focusout":
              S2 = "blur", g2 = ol;
              break;
            case "beforeblur":
            case "afterblur":
              g2 = ol;
              break;
            case "click":
              if (t.button === 2) break e;
            case "auxclick":
            case "dblclick":
            case "mousedown":
            case "mousemove":
            case "mouseup":
            case "mouseout":
            case "mouseover":
            case "contextmenu":
              g2 = Su;
              break;
            case "drag":
            case "dragend":
            case "dragenter":
            case "dragexit":
            case "dragleave":
            case "dragover":
            case "dragstart":
            case "drop":
              g2 = tc;
              break;
            case "touchcancel":
            case "touchend":
            case "touchmove":
            case "touchstart":
              g2 = yc;
              break;
            case ns:
            case ts:
            case rs:
              g2 = ic;
              break;
            case ls:
              g2 = wc;
              break;
            case "scroll":
              g2 = ec;
              break;
            case "wheel":
              g2 = kc;
              break;
            case "copy":
            case "cut":
            case "paste":
              g2 = oc;
              break;
            case "gotpointercapture":
            case "lostpointercapture":
            case "pointercancel":
            case "pointerdown":
            case "pointermove":
            case "pointerout":
            case "pointerover":
            case "pointerup":
              g2 = Eu;
          }
          var k3 = (n2 & 4) !== 0, j2 = !k3 && e === "scroll", c3 = k3 ? p !== null ? p + "Capture" : null : p;
          k3 = [];
          for (var a2 = d3, f3; a2 !== null; ) {
            f3 = a2;
            var y3 = f3.stateNode;
            if (f3.tag === 5 && y3 !== null && (f3 = y3, c3 !== null && (y3 = St(a2, c3), y3 != null && k3.push(zt(a2, y3, f3)))), j2) break;
            a2 = a2.return;
          }
          0 < k3.length && (p = new g2(p, S2, null, t, m2), h3.push({ event: p, listeners: k3 }));
        }
      }
      if ((n2 & 7) === 0) {
        e: {
          if (p = e === "mouseover" || e === "pointerover", g2 = e === "mouseout" || e === "pointerout", p && t !== Rl && (S2 = t.relatedTarget || t.fromElement) && (cn(S2) || S2[Ie3])) break e;
          if ((g2 || p) && (p = m2.window === m2 ? m2 : (p = m2.ownerDocument) ? p.defaultView || p.parentWindow : globalThis, g2 ? (S2 = t.relatedTarget || t.toElement, g2 = d3, S2 = S2 ? cn(S2) : null, S2 !== null && (j2 = kn(S2), S2 !== j2 || S2.tag !== 5 && S2.tag !== 6) && (S2 = null)) : (g2 = null, S2 = d3), g2 !== S2)) {
            if (k3 = Su, y3 = "onMouseLeave", c3 = "onMouseEnter", a2 = "mouse", (e === "pointerout" || e === "pointerover") && (k3 = Eu, y3 = "onPointerLeave", c3 = "onPointerEnter", a2 = "pointer"), j2 = g2 == null ? p : Ln(g2), f3 = S2 == null ? p : Ln(S2), p = new k3(y3, a2 + "leave", g2, t, m2), p.target = j2, p.relatedTarget = f3, y3 = null, cn(m2) === d3 && (k3 = new k3(c3, a2 + "enter", S2, t, m2), k3.target = f3, k3.relatedTarget = j2, y3 = k3), j2 = y3, g2 && S2) n: {
              for (k3 = g2, c3 = S2, a2 = 0, f3 = k3; f3; f3 = Cn(f3)) a2++;
              for (f3 = 0, y3 = c3; y3; y3 = Cn(y3)) f3++;
              for (; 0 < a2 - f3; ) k3 = Cn(k3), a2--;
              for (; 0 < f3 - a2; ) c3 = Cn(c3), f3--;
              for (; a2--; ) {
                if (k3 === c3 || c3 !== null && k3 === c3.alternate) break n;
                k3 = Cn(k3), c3 = Cn(c3);
              }
              k3 = null;
            }
            else k3 = null;
            g2 !== null && Ru(h3, p, g2, k3, false), S2 !== null && j2 !== null && Ru(h3, j2, S2, k3, true);
          }
        }
        e: {
          if (p = d3 ? Ln(d3) : globalThis, g2 = p.nodeName && p.nodeName.toLowerCase(), g2 === "select" || g2 === "input" && p.type === "file") var E3 = Pc;
          else if (Nu(p)) if (Zo) E3 = Dc;
          else {
            E3 = Tc;
            var C = Lc;
          }
          else (g2 = p.nodeName) && g2.toLowerCase() === "input" && (p.type === "checkbox" || p.type === "radio") && (E3 = Mc);
          if (E3 && (E3 = E3(e, d3))) {
            Go(h3, E3, t, m2);
            break e;
          }
          C && C(e, p, d3), e === "focusout" && (C = p._wrapperState) && C.controlled && p.type === "number" && Ll(p, "number", p.value);
        }
        switch (C = d3 ? Ln(d3) : globalThis, e) {
          case "focusin":
            (Nu(C) || C.contentEditable === "true") && (zn = C, Bl = d3, pt = null);
            break;
          case "focusout":
            pt = Bl = zn = null;
            break;
          case "mousedown":
            Hl = true;
            break;
          case "contextmenu":
          case "mouseup":
          case "dragend":
            Hl = false, Lu(h3, t, m2);
            break;
          case "selectionchange":
            if (Fc) break;
          case "keydown":
          case "keyup":
            Lu(h3, t, m2);
        }
        var x4;
        if (Pi) e: {
          switch (e) {
            case "compositionstart":
              var N = "onCompositionStart";
              break e;
            case "compositionend":
              N = "onCompositionEnd";
              break e;
            case "compositionupdate":
              N = "onCompositionUpdate";
              break e;
          }
          N = void 0;
        }
        else _n ? Yo(e, t) && (N = "onCompositionEnd") : e === "keydown" && t.keyCode === 229 && (N = "onCompositionStart");
        N && (Ko && t.locale !== "ko" && (_n || N !== "onCompositionStart" ? N === "onCompositionEnd" && _n && (x4 = $o()) : (Ke = m2, Ni = "value" in Ke ? Ke.value : Ke.textContent, _n = true)), C = kr(d3, N), 0 < C.length && (N = new ku(N, e, null, t, m2), h3.push({ event: N, listeners: C }), x4 ? N.data = x4 : (x4 = Xo(t), x4 !== null && (N.data = x4)))), (x4 = Cc ? xc(e, t) : Nc(e, t)) && (d3 = kr(d3, "onBeforeInput"), 0 < d3.length && (m2 = new ku("onBeforeInput", "beforeinput", null, t, m2), h3.push({ event: m2, listeners: d3 }), m2.data = x4));
      }
      us(h3, n2);
    });
  }
  function zt(e, n2, t) {
    return { instance: e, listener: n2, currentTarget: t };
  }
  function kr(e, n2) {
    for (var t = n2 + "Capture", r = []; e !== null; ) {
      var l3 = e, i2 = l3.stateNode;
      l3.tag === 5 && i2 !== null && (l3 = i2, i2 = St(e, t), i2 != null && r.unshift(zt(e, i2, l3)), i2 = St(e, n2), i2 != null && r.push(zt(e, i2, l3))), e = e.return;
    }
    return r;
  }
  function Cn(e) {
    if (e === null) return null;
    do
      e = e.return;
    while (e && e.tag !== 5);
    return e || null;
  }
  function Ru(e, n2, t, r, l3) {
    for (var i2 = n2._reactName, u2 = []; t !== null && t !== r; ) {
      var o = t, s = o.alternate, d3 = o.stateNode;
      if (s !== null && s === r) break;
      o.tag === 5 && d3 !== null && (o = d3, l3 ? (s = St(t, i2), s != null && u2.unshift(zt(t, s, o))) : l3 || (s = St(t, i2), s != null && u2.push(zt(t, s, o)))), t = t.return;
    }
    u2.length !== 0 && e.push({ event: n2, listeners: u2 });
  }
  var jc = /\r\n?/g, Uc = /\u0000|\uFFFD/g;
  function Fu(e) {
    return (typeof e == "string" ? e : "" + e).replace(jc, `
`).replace(Uc, "");
  }
  function qt(e, n2, t) {
    if (n2 = Fu(n2), Fu(e) !== n2 && t) throw Error(v2(425));
  }
  function Er() {
  }
  var Wl = null, Ql = null;
  function $l(e, n2) {
    return e === "textarea" || e === "noscript" || typeof n2.children == "string" || typeof n2.children == "number" || typeof n2.dangerouslySetInnerHTML == "object" && n2.dangerouslySetInnerHTML !== null && n2.dangerouslySetInnerHTML.__html != null;
  }
  var Kl = typeof setTimeout == "function" ? setTimeout : void 0, Vc = typeof clearTimeout == "function" ? clearTimeout : void 0, Iu = typeof Promise == "function" ? Promise : void 0, Ac = typeof queueMicrotask == "function" ? queueMicrotask : typeof Iu < "u" ? function(e) {
    return Iu.resolve(null).then(e).catch(Bc);
  } : Kl;
  function Bc(e) {
    setTimeout(function() {
      throw e;
    });
  }
  function dl(e, n2) {
    var t = n2, r = 0;
    do {
      var l3 = t.nextSibling;
      if (e.removeChild(t), l3 && l3.nodeType === 8) if (t = l3.data, t === "/$") {
        if (r === 0) {
          e.removeChild(l3), Ct(n2);
          return;
        }
        r--;
      } else t !== "$" && t !== "$?" && t !== "$!" || r++;
      t = l3;
    } while (t);
    Ct(n2);
  }
  function Je(e) {
    for (; e != null; e = e.nextSibling) {
      var n2 = e.nodeType;
      if (n2 === 1 || n2 === 3) break;
      if (n2 === 8) {
        if (n2 = e.data, n2 === "$" || n2 === "$!" || n2 === "$?") break;
        if (n2 === "/$") return null;
      }
    }
    return e;
  }
  function ju(e) {
    e = e.previousSibling;
    for (var n2 = 0; e; ) {
      if (e.nodeType === 8) {
        var t = e.data;
        if (t === "$" || t === "$!" || t === "$?") {
          if (n2 === 0) return e;
          n2--;
        } else t === "/$" && n2++;
      }
      e = e.previousSibling;
    }
    return null;
  }
  var Zn = Math.random().toString(36).slice(2), ze2 = "__reactFiber$" + Zn, Pt = "__reactProps$" + Zn, Ie3 = "__reactContainer$" + Zn, Yl = "__reactEvents$" + Zn, Hc = "__reactListeners$" + Zn, Wc = "__reactHandles$" + Zn;
  function cn(e) {
    var n2 = e[ze2];
    if (n2) return n2;
    for (var t = e.parentNode; t; ) {
      if (n2 = t[Ie3] || t[ze2]) {
        if (t = n2.alternate, n2.child !== null || t !== null && t.child !== null) for (e = ju(e); e !== null; ) {
          if (t = e[ze2]) return t;
          e = ju(e);
        }
        return n2;
      }
      e = t, t = e.parentNode;
    }
    return null;
  }
  function jt(e) {
    return e = e[ze2] || e[Ie3], !e || e.tag !== 5 && e.tag !== 6 && e.tag !== 13 && e.tag !== 3 ? null : e;
  }
  function Ln(e) {
    if (e.tag === 5 || e.tag === 6) return e.stateNode;
    throw Error(v2(33));
  }
  function Hr(e) {
    return e[Pt] || null;
  }
  var Xl = [], Tn = -1;
  function un(e) {
    return { current: e };
  }
  function M2(e) {
    0 > Tn || (e.current = Xl[Tn], Xl[Tn] = null, Tn--);
  }
  function L3(e, n2) {
    Tn++, Xl[Tn] = e.current, e.current = n2;
  }
  var rn = {}, J = un(rn), re = un(false), hn = rn;
  function Wn(e, n2) {
    var t = e.type.contextTypes;
    if (!t) return rn;
    var r = e.stateNode;
    if (r && r.__reactInternalMemoizedUnmaskedChildContext === n2) return r.__reactInternalMemoizedMaskedChildContext;
    var l3 = {}, i2;
    for (i2 in t) l3[i2] = n2[i2];
    return r && (e = e.stateNode, e.__reactInternalMemoizedUnmaskedChildContext = n2, e.__reactInternalMemoizedMaskedChildContext = l3), l3;
  }
  function le2(e) {
    return e = e.childContextTypes, e != null;
  }
  function Cr() {
    M2(re), M2(J);
  }
  function Uu(e, n2, t) {
    if (J.current !== rn) throw Error(v2(168));
    L3(J, n2), L3(re, t);
  }
  function ss(e, n2, t) {
    var r = e.stateNode;
    if (n2 = n2.childContextTypes, typeof r.getChildContext != "function") return t;
    r = r.getChildContext();
    for (var l3 in r) if (!(l3 in n2)) throw Error(v2(108, La(e) || "Unknown", l3));
    return F3({}, t, r);
  }
  function xr(e) {
    return e = (e = e.stateNode) && e.__reactInternalMemoizedMergedChildContext || rn, hn = J.current, L3(J, e), L3(re, re.current), true;
  }
  function Vu(e, n2, t) {
    var r = e.stateNode;
    if (!r) throw Error(v2(169));
    t ? (e = ss(e, n2, hn), r.__reactInternalMemoizedMergedChildContext = e, M2(re), M2(J), L3(J, e)) : M2(re), L3(re, t);
  }
  var Me2 = null, Wr = false, pl = false;
  function as(e) {
    Me2 === null ? Me2 = [e] : Me2.push(e);
  }
  function Qc(e) {
    Wr = true, as(e);
  }
  function on() {
    if (!pl && Me2 !== null) {
      pl = true;
      var e = 0, n2 = P;
      try {
        var t = Me2;
        for (P = 1; e < t.length; e++) {
          var r = t[e];
          do
            r = r(true);
          while (r !== null);
        }
        Me2 = null, Wr = false;
      } catch (l3) {
        throw Me2 !== null && (Me2 = Me2.slice(e + 1)), Ro(ki, on), l3;
      } finally {
        P = n2, pl = false;
      }
    }
    return null;
  }
  var Mn = [], Dn = 0, Nr = null, _r = 0, de3 = [], pe3 = 0, vn = null, De2 = 1, Oe2 = "";
  function sn(e, n2) {
    Mn[Dn++] = _r, Mn[Dn++] = Nr, Nr = e, _r = n2;
  }
  function cs(e, n2, t) {
    de3[pe3++] = De2, de3[pe3++] = Oe2, de3[pe3++] = vn, vn = e;
    var r = De2;
    e = Oe2;
    var l3 = 32 - Ee2(r) - 1;
    r &= ~(1 << l3), t += 1;
    var i2 = 32 - Ee2(n2) + l3;
    if (30 < i2) {
      var u2 = l3 - l3 % 5;
      i2 = (r & (1 << u2) - 1).toString(32), r >>= u2, l3 -= u2, De2 = 1 << 32 - Ee2(n2) + l3 | t << l3 | r, Oe2 = i2 + e;
    } else De2 = 1 << i2 | t << l3 | r, Oe2 = e;
  }
  function Ti(e) {
    e.return !== null && (sn(e, 1), cs(e, 1, 0));
  }
  function Mi(e) {
    for (; e === Nr; ) Nr = Mn[--Dn], Mn[Dn] = null, _r = Mn[--Dn], Mn[Dn] = null;
    for (; e === vn; ) vn = de3[--pe3], de3[pe3] = null, Oe2 = de3[--pe3], de3[pe3] = null, De2 = de3[--pe3], de3[pe3] = null;
  }
  var se2 = null, oe2 = null, D2 = false, ke3 = null;
  function fs(e, n2) {
    var t = me3(5, null, null, 0);
    t.elementType = "DELETED", t.stateNode = n2, t.return = e, n2 = e.deletions, n2 === null ? (e.deletions = [t], e.flags |= 16) : n2.push(t);
  }
  function Au(e, n2) {
    switch (e.tag) {
      case 5:
        var t = e.type;
        return n2 = n2.nodeType !== 1 || t.toLowerCase() !== n2.nodeName.toLowerCase() ? null : n2, n2 !== null ? (e.stateNode = n2, se2 = e, oe2 = Je(n2.firstChild), true) : false;
      case 6:
        return n2 = e.pendingProps === "" || n2.nodeType !== 3 ? null : n2, n2 !== null ? (e.stateNode = n2, se2 = e, oe2 = null, true) : false;
      case 13:
        return n2 = n2.nodeType !== 8 ? null : n2, n2 !== null ? (t = vn !== null ? { id: De2, overflow: Oe2 } : null, e.memoizedState = { dehydrated: n2, treeContext: t, retryLane: 1073741824 }, t = me3(18, null, null, 0), t.stateNode = n2, t.return = e, e.child = t, se2 = e, oe2 = null, true) : false;
      default:
        return false;
    }
  }
  function Gl(e) {
    return (e.mode & 1) !== 0 && (e.flags & 128) === 0;
  }
  function Zl(e) {
    if (D2) {
      var n2 = oe2;
      if (n2) {
        var t = n2;
        if (!Au(e, n2)) {
          if (Gl(e)) throw Error(v2(418));
          n2 = Je(t.nextSibling);
          var r = se2;
          n2 && Au(e, n2) ? fs(r, t) : (e.flags = e.flags & -4097 | 2, D2 = false, se2 = e);
        }
      } else {
        if (Gl(e)) throw Error(v2(418));
        e.flags = e.flags & -4097 | 2, D2 = false, se2 = e;
      }
    }
  }
  function Bu(e) {
    for (e = e.return; e !== null && e.tag !== 5 && e.tag !== 3 && e.tag !== 13; ) e = e.return;
    se2 = e;
  }
  function bt(e) {
    if (e !== se2) return false;
    if (!D2) return Bu(e), D2 = true, false;
    var n2;
    if ((n2 = e.tag !== 3) && !(n2 = e.tag !== 5) && (n2 = e.type, n2 = n2 !== "head" && n2 !== "body" && !$l(e.type, e.memoizedProps)), n2 && (n2 = oe2)) {
      if (Gl(e)) throw ds(), Error(v2(418));
      for (; n2; ) fs(e, n2), n2 = Je(n2.nextSibling);
    }
    if (Bu(e), e.tag === 13) {
      if (e = e.memoizedState, e = e !== null ? e.dehydrated : null, !e) throw Error(v2(317));
      e: {
        for (e = e.nextSibling, n2 = 0; e; ) {
          if (e.nodeType === 8) {
            var t = e.data;
            if (t === "/$") {
              if (n2 === 0) {
                oe2 = Je(e.nextSibling);
                break e;
              }
              n2--;
            } else t !== "$" && t !== "$!" && t !== "$?" || n2++;
          }
          e = e.nextSibling;
        }
        oe2 = null;
      }
    } else oe2 = se2 ? Je(e.stateNode.nextSibling) : null;
    return true;
  }
  function ds() {
    for (var e = oe2; e; ) e = Je(e.nextSibling);
  }
  function Qn() {
    oe2 = se2 = null, D2 = false;
  }
  function Di(e) {
    ke3 === null ? ke3 = [e] : ke3.push(e);
  }
  var $c = Ve2.ReactCurrentBatchConfig;
  function nt(e, n2, t) {
    if (e = t.ref, e !== null && typeof e != "function" && typeof e != "object") {
      if (t._owner) {
        if (t = t._owner, t) {
          if (t.tag !== 1) throw Error(v2(309));
          var r = t.stateNode;
        }
        if (!r) throw Error(v2(147, e));
        var l3 = r, i2 = "" + e;
        return n2 !== null && n2.ref !== null && typeof n2.ref == "function" && n2.ref._stringRef === i2 ? n2.ref : (n2 = function(u2) {
          var o = l3.refs;
          u2 === null ? delete o[i2] : o[i2] = u2;
        }, n2._stringRef = i2, n2);
      }
      if (typeof e != "string") throw Error(v2(284));
      if (!t._owner) throw Error(v2(290, e));
    }
    return e;
  }
  function er(e, n2) {
    throw e = Object.prototype.toString.call(n2), Error(v2(31, e === "[object Object]" ? "object with keys {" + Object.keys(n2).join(", ") + "}" : e));
  }
  function Hu(e) {
    var n2 = e._init;
    return n2(e._payload);
  }
  function ps(e) {
    function n2(c3, a2) {
      if (e) {
        var f3 = c3.deletions;
        f3 === null ? (c3.deletions = [a2], c3.flags |= 16) : f3.push(a2);
      }
    }
    function t(c3, a2) {
      if (!e) return null;
      for (; a2 !== null; ) n2(c3, a2), a2 = a2.sibling;
      return null;
    }
    function r(c3, a2) {
      for (c3 = /* @__PURE__ */ new Map(); a2 !== null; ) a2.key !== null ? c3.set(a2.key, a2) : c3.set(a2.index, a2), a2 = a2.sibling;
      return c3;
    }
    function l3(c3, a2) {
      return c3 = nn(c3, a2), c3.index = 0, c3.sibling = null, c3;
    }
    function i2(c3, a2, f3) {
      return c3.index = f3, e ? (f3 = c3.alternate, f3 !== null ? (f3 = f3.index, f3 < a2 ? (c3.flags |= 2, a2) : f3) : (c3.flags |= 2, a2)) : (c3.flags |= 1048576, a2);
    }
    function u2(c3) {
      return e && c3.alternate === null && (c3.flags |= 2), c3;
    }
    function o(c3, a2, f3, y3) {
      return a2 === null || a2.tag !== 6 ? (a2 = Sl(f3, c3.mode, y3), a2.return = c3, a2) : (a2 = l3(a2, f3), a2.return = c3, a2);
    }
    function s(c3, a2, f3, y3) {
      var E3 = f3.type;
      return E3 === Nn ? m2(c3, a2, f3.props.children, y3, f3.key) : a2 !== null && (a2.elementType === E3 || typeof E3 == "object" && E3 !== null && E3.$$typeof === He2 && Hu(E3) === a2.type) ? (y3 = l3(a2, f3.props), y3.ref = nt(c3, a2, f3), y3.return = c3, y3) : (y3 = pr(f3.type, f3.key, f3.props, null, c3.mode, y3), y3.ref = nt(c3, a2, f3), y3.return = c3, y3);
    }
    function d3(c3, a2, f3, y3) {
      return a2 === null || a2.tag !== 4 || a2.stateNode.containerInfo !== f3.containerInfo || a2.stateNode.implementation !== f3.implementation ? (a2 = kl(f3, c3.mode, y3), a2.return = c3, a2) : (a2 = l3(a2, f3.children || []), a2.return = c3, a2);
    }
    function m2(c3, a2, f3, y3, E3) {
      return a2 === null || a2.tag !== 7 ? (a2 = mn(f3, c3.mode, y3, E3), a2.return = c3, a2) : (a2 = l3(a2, f3), a2.return = c3, a2);
    }
    function h3(c3, a2, f3) {
      if (typeof a2 == "string" && a2 !== "" || typeof a2 == "number") return a2 = Sl("" + a2, c3.mode, f3), a2.return = c3, a2;
      if (typeof a2 == "object" && a2 !== null) {
        switch (a2.$$typeof) {
          case Vt:
            return f3 = pr(a2.type, a2.key, a2.props, null, c3.mode, f3), f3.ref = nt(c3, null, a2), f3.return = c3, f3;
          case xn:
            return a2 = kl(a2, c3.mode, f3), a2.return = c3, a2;
          case He2:
            var y3 = a2._init;
            return h3(c3, y3(a2._payload), f3);
        }
        if (ut(a2) || Jn(a2)) return a2 = mn(a2, c3.mode, f3, null), a2.return = c3, a2;
        er(c3, a2);
      }
      return null;
    }
    function p(c3, a2, f3, y3) {
      var E3 = a2 !== null ? a2.key : null;
      if (typeof f3 == "string" && f3 !== "" || typeof f3 == "number") return E3 !== null ? null : o(c3, a2, "" + f3, y3);
      if (typeof f3 == "object" && f3 !== null) {
        switch (f3.$$typeof) {
          case Vt:
            return f3.key === E3 ? s(c3, a2, f3, y3) : null;
          case xn:
            return f3.key === E3 ? d3(c3, a2, f3, y3) : null;
          case He2:
            return E3 = f3._init, p(c3, a2, E3(f3._payload), y3);
        }
        if (ut(f3) || Jn(f3)) return E3 !== null ? null : m2(c3, a2, f3, y3, null);
        er(c3, f3);
      }
      return null;
    }
    function g2(c3, a2, f3, y3, E3) {
      if (typeof y3 == "string" && y3 !== "" || typeof y3 == "number") return c3 = c3.get(f3) || null, o(a2, c3, "" + y3, E3);
      if (typeof y3 == "object" && y3 !== null) {
        switch (y3.$$typeof) {
          case Vt:
            return c3 = c3.get(y3.key === null ? f3 : y3.key) || null, s(a2, c3, y3, E3);
          case xn:
            return c3 = c3.get(y3.key === null ? f3 : y3.key) || null, d3(a2, c3, y3, E3);
          case He2:
            var C = y3._init;
            return g2(c3, a2, f3, C(y3._payload), E3);
        }
        if (ut(y3) || Jn(y3)) return c3 = c3.get(f3) || null, m2(a2, c3, y3, E3, null);
        er(a2, y3);
      }
      return null;
    }
    function S2(c3, a2, f3, y3) {
      for (var E3 = null, C = null, x4 = a2, N = a2 = 0, H2 = null; x4 !== null && N < f3.length; N++) {
        x4.index > N ? (H2 = x4, x4 = null) : H2 = x4.sibling;
        var z2 = p(c3, x4, f3[N], y3);
        if (z2 === null) {
          x4 === null && (x4 = H2);
          break;
        }
        e && x4 && z2.alternate === null && n2(c3, x4), a2 = i2(z2, a2, N), C === null ? E3 = z2 : C.sibling = z2, C = z2, x4 = H2;
      }
      if (N === f3.length) return t(c3, x4), D2 && sn(c3, N), E3;
      if (x4 === null) {
        for (; N < f3.length; N++) x4 = h3(c3, f3[N], y3), x4 !== null && (a2 = i2(x4, a2, N), C === null ? E3 = x4 : C.sibling = x4, C = x4);
        return D2 && sn(c3, N), E3;
      }
      for (x4 = r(c3, x4); N < f3.length; N++) H2 = g2(x4, c3, N, f3[N], y3), H2 !== null && (e && H2.alternate !== null && x4.delete(H2.key === null ? N : H2.key), a2 = i2(H2, a2, N), C === null ? E3 = H2 : C.sibling = H2, C = H2);
      return e && x4.forEach(function(Ae2) {
        return n2(c3, Ae2);
      }), D2 && sn(c3, N), E3;
    }
    function k3(c3, a2, f3, y3) {
      var E3 = Jn(f3);
      if (typeof E3 != "function") throw Error(v2(150));
      if (f3 = E3.call(f3), f3 == null) throw Error(v2(151));
      for (var C = E3 = null, x4 = a2, N = a2 = 0, H2 = null, z2 = f3.next(); x4 !== null && !z2.done; N++, z2 = f3.next()) {
        x4.index > N ? (H2 = x4, x4 = null) : H2 = x4.sibling;
        var Ae2 = p(c3, x4, z2.value, y3);
        if (Ae2 === null) {
          x4 === null && (x4 = H2);
          break;
        }
        e && x4 && Ae2.alternate === null && n2(c3, x4), a2 = i2(Ae2, a2, N), C === null ? E3 = Ae2 : C.sibling = Ae2, C = Ae2, x4 = H2;
      }
      if (z2.done) return t(c3, x4), D2 && sn(c3, N), E3;
      if (x4 === null) {
        for (; !z2.done; N++, z2 = f3.next()) z2 = h3(c3, z2.value, y3), z2 !== null && (a2 = i2(z2, a2, N), C === null ? E3 = z2 : C.sibling = z2, C = z2);
        return D2 && sn(c3, N), E3;
      }
      for (x4 = r(c3, x4); !z2.done; N++, z2 = f3.next()) z2 = g2(x4, c3, N, z2.value, y3), z2 !== null && (e && z2.alternate !== null && x4.delete(z2.key === null ? N : z2.key), a2 = i2(z2, a2, N), C === null ? E3 = z2 : C.sibling = z2, C = z2);
      return e && x4.forEach(function(ha) {
        return n2(c3, ha);
      }), D2 && sn(c3, N), E3;
    }
    function j2(c3, a2, f3, y3) {
      if (typeof f3 == "object" && f3 !== null && f3.type === Nn && f3.key === null && (f3 = f3.props.children), typeof f3 == "object" && f3 !== null) {
        switch (f3.$$typeof) {
          case Vt:
            e: {
              for (var E3 = f3.key, C = a2; C !== null; ) {
                if (C.key === E3) {
                  if (E3 = f3.type, E3 === Nn) {
                    if (C.tag === 7) {
                      t(c3, C.sibling), a2 = l3(C, f3.props.children), a2.return = c3, c3 = a2;
                      break e;
                    }
                  } else if (C.elementType === E3 || typeof E3 == "object" && E3 !== null && E3.$$typeof === He2 && Hu(E3) === C.type) {
                    t(c3, C.sibling), a2 = l3(C, f3.props), a2.ref = nt(c3, C, f3), a2.return = c3, c3 = a2;
                    break e;
                  }
                  t(c3, C);
                  break;
                } else n2(c3, C);
                C = C.sibling;
              }
              f3.type === Nn ? (a2 = mn(f3.props.children, c3.mode, y3, f3.key), a2.return = c3, c3 = a2) : (y3 = pr(f3.type, f3.key, f3.props, null, c3.mode, y3), y3.ref = nt(c3, a2, f3), y3.return = c3, c3 = y3);
            }
            return u2(c3);
          case xn:
            e: {
              for (C = f3.key; a2 !== null; ) {
                if (a2.key === C) if (a2.tag === 4 && a2.stateNode.containerInfo === f3.containerInfo && a2.stateNode.implementation === f3.implementation) {
                  t(c3, a2.sibling), a2 = l3(a2, f3.children || []), a2.return = c3, c3 = a2;
                  break e;
                } else {
                  t(c3, a2);
                  break;
                }
                else n2(c3, a2);
                a2 = a2.sibling;
              }
              a2 = kl(f3, c3.mode, y3), a2.return = c3, c3 = a2;
            }
            return u2(c3);
          case He2:
            return C = f3._init, j2(c3, a2, C(f3._payload), y3);
        }
        if (ut(f3)) return S2(c3, a2, f3, y3);
        if (Jn(f3)) return k3(c3, a2, f3, y3);
        er(c3, f3);
      }
      return typeof f3 == "string" && f3 !== "" || typeof f3 == "number" ? (f3 = "" + f3, a2 !== null && a2.tag === 6 ? (t(c3, a2.sibling), a2 = l3(a2, f3), a2.return = c3, c3 = a2) : (t(c3, a2), a2 = Sl(f3, c3.mode, y3), a2.return = c3, c3 = a2), u2(c3)) : t(c3, a2);
    }
    return j2;
  }
  var $n = ps(true), ms = ps(false), zr = un(null), Pr = null, On = null, Oi = null;
  function Ri() {
    Oi = On = Pr = null;
  }
  function Fi(e) {
    var n2 = zr.current;
    M2(zr), e._currentValue = n2;
  }
  function Jl(e, n2, t) {
    for (; e !== null; ) {
      var r = e.alternate;
      if ((e.childLanes & n2) !== n2 ? (e.childLanes |= n2, r !== null && (r.childLanes |= n2)) : r !== null && (r.childLanes & n2) !== n2 && (r.childLanes |= n2), e === t) break;
      e = e.return;
    }
  }
  function An(e, n2) {
    Pr = e, Oi = On = null, e = e.dependencies, e !== null && e.firstContext !== null && ((e.lanes & n2) !== 0 && (te = true), e.firstContext = null);
  }
  function ve3(e) {
    var n2 = e._currentValue;
    if (Oi !== e) if (e = { context: e, memoizedValue: n2, next: null }, On === null) {
      if (Pr === null) throw Error(v2(308));
      On = e, Pr.dependencies = { lanes: 0, firstContext: e };
    } else On = On.next = e;
    return n2;
  }
  var fn = null;
  function Ii(e) {
    fn === null ? fn = [e] : fn.push(e);
  }
  function hs(e, n2, t, r) {
    var l3 = n2.interleaved;
    return l3 === null ? (t.next = t, Ii(n2)) : (t.next = l3.next, l3.next = t), n2.interleaved = t, je2(e, r);
  }
  function je2(e, n2) {
    e.lanes |= n2;
    var t = e.alternate;
    for (t !== null && (t.lanes |= n2), t = e, e = e.return; e !== null; ) e.childLanes |= n2, t = e.alternate, t !== null && (t.childLanes |= n2), t = e, e = e.return;
    return t.tag === 3 ? t.stateNode : null;
  }
  var We2 = false;
  function ji(e) {
    e.updateQueue = { baseState: e.memoizedState, firstBaseUpdate: null, lastBaseUpdate: null, shared: { pending: null, interleaved: null, lanes: 0 }, effects: null };
  }
  function vs(e, n2) {
    e = e.updateQueue, n2.updateQueue === e && (n2.updateQueue = { baseState: e.baseState, firstBaseUpdate: e.firstBaseUpdate, lastBaseUpdate: e.lastBaseUpdate, shared: e.shared, effects: e.effects });
  }
  function Re2(e, n2) {
    return { eventTime: e, lane: n2, tag: 0, payload: null, callback: null, next: null };
  }
  function qe2(e, n2, t) {
    var r = e.updateQueue;
    if (r === null) return null;
    if (r = r.shared, (_2 & 2) !== 0) {
      var l3 = r.pending;
      return l3 === null ? n2.next = n2 : (n2.next = l3.next, l3.next = n2), r.pending = n2, je2(e, t);
    }
    return l3 = r.interleaved, l3 === null ? (n2.next = n2, Ii(r)) : (n2.next = l3.next, l3.next = n2), r.interleaved = n2, je2(e, t);
  }
  function or(e, n2, t) {
    if (n2 = n2.updateQueue, n2 !== null && (n2 = n2.shared, (t & 4194240) !== 0)) {
      var r = n2.lanes;
      r &= e.pendingLanes, t |= r, n2.lanes = t, Ei(e, t);
    }
  }
  function Wu(e, n2) {
    var t = e.updateQueue, r = e.alternate;
    if (r !== null && (r = r.updateQueue, t === r)) {
      var l3 = null, i2 = null;
      if (t = t.firstBaseUpdate, t !== null) {
        do {
          var u2 = { eventTime: t.eventTime, lane: t.lane, tag: t.tag, payload: t.payload, callback: t.callback, next: null };
          i2 === null ? l3 = i2 = u2 : i2 = i2.next = u2, t = t.next;
        } while (t !== null);
        i2 === null ? l3 = i2 = n2 : i2 = i2.next = n2;
      } else l3 = i2 = n2;
      t = { baseState: r.baseState, firstBaseUpdate: l3, lastBaseUpdate: i2, shared: r.shared, effects: r.effects }, e.updateQueue = t;
      return;
    }
    e = t.lastBaseUpdate, e === null ? t.firstBaseUpdate = n2 : e.next = n2, t.lastBaseUpdate = n2;
  }
  function Lr(e, n2, t, r) {
    var l3 = e.updateQueue;
    We2 = false;
    var i2 = l3.firstBaseUpdate, u2 = l3.lastBaseUpdate, o = l3.shared.pending;
    if (o !== null) {
      l3.shared.pending = null;
      var s = o, d3 = s.next;
      s.next = null, u2 === null ? i2 = d3 : u2.next = d3, u2 = s;
      var m2 = e.alternate;
      m2 !== null && (m2 = m2.updateQueue, o = m2.lastBaseUpdate, o !== u2 && (o === null ? m2.firstBaseUpdate = d3 : o.next = d3, m2.lastBaseUpdate = s));
    }
    if (i2 !== null) {
      var h3 = l3.baseState;
      u2 = 0, m2 = d3 = s = null, o = i2;
      do {
        var p = o.lane, g2 = o.eventTime;
        if ((r & p) === p) {
          m2 !== null && (m2 = m2.next = { eventTime: g2, lane: 0, tag: o.tag, payload: o.payload, callback: o.callback, next: null });
          e: {
            var S2 = e, k3 = o;
            switch (p = n2, g2 = t, k3.tag) {
              case 1:
                if (S2 = k3.payload, typeof S2 == "function") {
                  h3 = S2.call(g2, h3, p);
                  break e;
                }
                h3 = S2;
                break e;
              case 3:
                S2.flags = S2.flags & -65537 | 128;
              case 0:
                if (S2 = k3.payload, p = typeof S2 == "function" ? S2.call(g2, h3, p) : S2, p == null) break e;
                h3 = F3({}, h3, p);
                break e;
              case 2:
                We2 = true;
            }
          }
          o.callback !== null && o.lane !== 0 && (e.flags |= 64, p = l3.effects, p === null ? l3.effects = [o] : p.push(o));
        } else g2 = { eventTime: g2, lane: p, tag: o.tag, payload: o.payload, callback: o.callback, next: null }, m2 === null ? (d3 = m2 = g2, s = h3) : m2 = m2.next = g2, u2 |= p;
        if (o = o.next, o === null) {
          if (o = l3.shared.pending, o === null) break;
          p = o, o = p.next, p.next = null, l3.lastBaseUpdate = p, l3.shared.pending = null;
        }
      } while (true);
      if (m2 === null && (s = h3), l3.baseState = s, l3.firstBaseUpdate = d3, l3.lastBaseUpdate = m2, n2 = l3.shared.interleaved, n2 !== null) {
        l3 = n2;
        do
          u2 |= l3.lane, l3 = l3.next;
        while (l3 !== n2);
      } else i2 === null && (l3.shared.lanes = 0);
      gn |= u2, e.lanes = u2, e.memoizedState = h3;
    }
  }
  function Qu(e, n2, t) {
    if (e = n2.effects, n2.effects = null, e !== null) for (n2 = 0; n2 < e.length; n2++) {
      var r = e[n2], l3 = r.callback;
      if (l3 !== null) {
        if (r.callback = null, r = t, typeof l3 != "function") throw Error(v2(191, l3));
        l3.call(r);
      }
    }
  }
  var Ut = {}, Le2 = un(Ut), Lt = un(Ut), Tt = un(Ut);
  function dn(e) {
    if (e === Ut) throw Error(v2(174));
    return e;
  }
  function Ui(e, n2) {
    switch (L3(Tt, n2), L3(Lt, e), L3(Le2, Ut), e = n2.nodeType, e) {
      case 9:
      case 11:
        n2 = (n2 = n2.documentElement) ? n2.namespaceURI : Ml(null, "");
        break;
      default:
        e = e === 8 ? n2.parentNode : n2, n2 = e.namespaceURI || null, e = e.tagName, n2 = Ml(n2, e);
    }
    M2(Le2), L3(Le2, n2);
  }
  function Kn() {
    M2(Le2), M2(Lt), M2(Tt);
  }
  function ys(e) {
    dn(Tt.current);
    var n2 = dn(Le2.current), t = Ml(n2, e.type);
    n2 !== t && (L3(Lt, e), L3(Le2, t));
  }
  function Vi(e) {
    Lt.current === e && (M2(Le2), M2(Lt));
  }
  var O3 = un(0);
  function Tr(e) {
    for (var n2 = e; n2 !== null; ) {
      if (n2.tag === 13) {
        var t = n2.memoizedState;
        if (t !== null && (t = t.dehydrated, t === null || t.data === "$?" || t.data === "$!")) return n2;
      } else if (n2.tag === 19 && n2.memoizedProps.revealOrder !== void 0) {
        if ((n2.flags & 128) !== 0) return n2;
      } else if (n2.child !== null) {
        n2.child.return = n2, n2 = n2.child;
        continue;
      }
      if (n2 === e) break;
      for (; n2.sibling === null; ) {
        if (n2.return === null || n2.return === e) return null;
        n2 = n2.return;
      }
      n2.sibling.return = n2.return, n2 = n2.sibling;
    }
    return null;
  }
  var ml = [];
  function Ai() {
    for (var e = 0; e < ml.length; e++) ml[e]._workInProgressVersionPrimary = null;
    ml.length = 0;
  }
  var sr = Ve2.ReactCurrentDispatcher, hl = Ve2.ReactCurrentBatchConfig, yn = 0, R2 = null, A2 = null, W = null, Mr = false, mt = false, Mt = 0, Kc = 0;
  function X2() {
    throw Error(v2(321));
  }
  function Bi(e, n2) {
    if (n2 === null) return false;
    for (var t = 0; t < n2.length && t < e.length; t++) if (!xe3(e[t], n2[t])) return false;
    return true;
  }
  function Hi(e, n2, t, r, l3, i2) {
    if (yn = i2, R2 = n2, n2.memoizedState = null, n2.updateQueue = null, n2.lanes = 0, sr.current = e === null || e.memoizedState === null ? Zc : Jc, e = t(r, l3), mt) {
      i2 = 0;
      do {
        if (mt = false, Mt = 0, 25 <= i2) throw Error(v2(301));
        i2 += 1, W = A2 = null, n2.updateQueue = null, sr.current = qc, e = t(r, l3);
      } while (mt);
    }
    if (sr.current = Dr, n2 = A2 !== null && A2.next !== null, yn = 0, W = A2 = R2 = null, Mr = false, n2) throw Error(v2(300));
    return e;
  }
  function Wi() {
    var e = Mt !== 0;
    return Mt = 0, e;
  }
  function _e3() {
    var e = { memoizedState: null, baseState: null, baseQueue: null, queue: null, next: null };
    return W === null ? R2.memoizedState = W = e : W = W.next = e, W;
  }
  function ye3() {
    if (A2 === null) {
      var e = R2.alternate;
      e = e !== null ? e.memoizedState : null;
    } else e = A2.next;
    var n2 = W === null ? R2.memoizedState : W.next;
    if (n2 !== null) W = n2, A2 = e;
    else {
      if (e === null) throw Error(v2(310));
      A2 = e, e = { memoizedState: A2.memoizedState, baseState: A2.baseState, baseQueue: A2.baseQueue, queue: A2.queue, next: null }, W === null ? R2.memoizedState = W = e : W = W.next = e;
    }
    return W;
  }
  function Dt(e, n2) {
    return typeof n2 == "function" ? n2(e) : n2;
  }
  function vl(e) {
    var n2 = ye3(), t = n2.queue;
    if (t === null) throw Error(v2(311));
    t.lastRenderedReducer = e;
    var r = A2, l3 = r.baseQueue, i2 = t.pending;
    if (i2 !== null) {
      if (l3 !== null) {
        var u2 = l3.next;
        l3.next = i2.next, i2.next = u2;
      }
      r.baseQueue = l3 = i2, t.pending = null;
    }
    if (l3 !== null) {
      i2 = l3.next, r = r.baseState;
      var o = u2 = null, s = null, d3 = i2;
      do {
        var m2 = d3.lane;
        if ((yn & m2) === m2) s !== null && (s = s.next = { lane: 0, action: d3.action, hasEagerState: d3.hasEagerState, eagerState: d3.eagerState, next: null }), r = d3.hasEagerState ? d3.eagerState : e(r, d3.action);
        else {
          var h3 = { lane: m2, action: d3.action, hasEagerState: d3.hasEagerState, eagerState: d3.eagerState, next: null };
          s === null ? (o = s = h3, u2 = r) : s = s.next = h3, R2.lanes |= m2, gn |= m2;
        }
        d3 = d3.next;
      } while (d3 !== null && d3 !== i2);
      s === null ? u2 = r : s.next = o, xe3(r, n2.memoizedState) || (te = true), n2.memoizedState = r, n2.baseState = u2, n2.baseQueue = s, t.lastRenderedState = r;
    }
    if (e = t.interleaved, e !== null) {
      l3 = e;
      do
        i2 = l3.lane, R2.lanes |= i2, gn |= i2, l3 = l3.next;
      while (l3 !== e);
    } else l3 === null && (t.lanes = 0);
    return [n2.memoizedState, t.dispatch];
  }
  function yl(e) {
    var n2 = ye3(), t = n2.queue;
    if (t === null) throw Error(v2(311));
    t.lastRenderedReducer = e;
    var r = t.dispatch, l3 = t.pending, i2 = n2.memoizedState;
    if (l3 !== null) {
      t.pending = null;
      var u2 = l3 = l3.next;
      do
        i2 = e(i2, u2.action), u2 = u2.next;
      while (u2 !== l3);
      xe3(i2, n2.memoizedState) || (te = true), n2.memoizedState = i2, n2.baseQueue === null && (n2.baseState = i2), t.lastRenderedState = i2;
    }
    return [i2, r];
  }
  function gs() {
  }
  function ws(e, n2) {
    var t = R2, r = ye3(), l3 = n2(), i2 = !xe3(r.memoizedState, l3);
    if (i2 && (r.memoizedState = l3, te = true), r = r.queue, Qi(Es.bind(null, t, r, e), [e]), r.getSnapshot !== n2 || i2 || W !== null && W.memoizedState.tag & 1) {
      if (t.flags |= 2048, Ot(9, ks.bind(null, t, r, l3, n2), void 0, null), Q === null) throw Error(v2(349));
      (yn & 30) !== 0 || Ss(t, n2, l3);
    }
    return l3;
  }
  function Ss(e, n2, t) {
    e.flags |= 16384, e = { getSnapshot: n2, value: t }, n2 = R2.updateQueue, n2 === null ? (n2 = { lastEffect: null, stores: null }, R2.updateQueue = n2, n2.stores = [e]) : (t = n2.stores, t === null ? n2.stores = [e] : t.push(e));
  }
  function ks(e, n2, t, r) {
    n2.value = t, n2.getSnapshot = r, Cs(n2) && xs(e);
  }
  function Es(e, n2, t) {
    return t(function() {
      Cs(n2) && xs(e);
    });
  }
  function Cs(e) {
    var n2 = e.getSnapshot;
    e = e.value;
    try {
      var t = n2();
      return !xe3(e, t);
    } catch {
      return true;
    }
  }
  function xs(e) {
    var n2 = je2(e, 1);
    n2 !== null && Ce2(n2, e, 1, -1);
  }
  function $u(e) {
    var n2 = _e3();
    return typeof e == "function" && (e = e()), n2.memoizedState = n2.baseState = e, e = { pending: null, interleaved: null, lanes: 0, dispatch: null, lastRenderedReducer: Dt, lastRenderedState: e }, n2.queue = e, e = e.dispatch = Gc.bind(null, R2, e), [n2.memoizedState, e];
  }
  function Ot(e, n2, t, r) {
    return e = { tag: e, create: n2, destroy: t, deps: r, next: null }, n2 = R2.updateQueue, n2 === null ? (n2 = { lastEffect: null, stores: null }, R2.updateQueue = n2, n2.lastEffect = e.next = e) : (t = n2.lastEffect, t === null ? n2.lastEffect = e.next = e : (r = t.next, t.next = e, e.next = r, n2.lastEffect = e)), e;
  }
  function Ns() {
    return ye3().memoizedState;
  }
  function ar(e, n2, t, r) {
    var l3 = _e3();
    R2.flags |= e, l3.memoizedState = Ot(1 | n2, t, void 0, r === void 0 ? null : r);
  }
  function Qr(e, n2, t, r) {
    var l3 = ye3();
    r = r === void 0 ? null : r;
    var i2 = void 0;
    if (A2 !== null) {
      var u2 = A2.memoizedState;
      if (i2 = u2.destroy, r !== null && Bi(r, u2.deps)) {
        l3.memoizedState = Ot(n2, t, i2, r);
        return;
      }
    }
    R2.flags |= e, l3.memoizedState = Ot(1 | n2, t, i2, r);
  }
  function Ku(e, n2) {
    return ar(8390656, 8, e, n2);
  }
  function Qi(e, n2) {
    return Qr(2048, 8, e, n2);
  }
  function _s(e, n2) {
    return Qr(4, 2, e, n2);
  }
  function zs(e, n2) {
    return Qr(4, 4, e, n2);
  }
  function Ps(e, n2) {
    if (typeof n2 == "function") return e = e(), n2(e), function() {
      n2(null);
    };
    if (n2 != null) return e = e(), n2.current = e, function() {
      n2.current = null;
    };
  }
  function Ls(e, n2, t) {
    return t = t != null ? t.concat([e]) : null, Qr(4, 4, Ps.bind(null, n2, e), t);
  }
  function $i() {
  }
  function Ts(e, n2) {
    var t = ye3();
    n2 = n2 === void 0 ? null : n2;
    var r = t.memoizedState;
    return r !== null && n2 !== null && Bi(n2, r[1]) ? r[0] : (t.memoizedState = [e, n2], e);
  }
  function Ms(e, n2) {
    var t = ye3();
    n2 = n2 === void 0 ? null : n2;
    var r = t.memoizedState;
    return r !== null && n2 !== null && Bi(n2, r[1]) ? r[0] : (e = e(), t.memoizedState = [e, n2], e);
  }
  function Ds(e, n2, t) {
    return (yn & 21) === 0 ? (e.baseState && (e.baseState = false, te = true), e.memoizedState = t) : (xe3(t, n2) || (t = jo(), R2.lanes |= t, gn |= t, e.baseState = true), n2);
  }
  function Yc(e, n2) {
    var t = P;
    P = t !== 0 && 4 > t ? t : 4, e(true);
    var r = hl.transition;
    hl.transition = {};
    try {
      e(false), n2();
    } finally {
      P = t, hl.transition = r;
    }
  }
  function Os() {
    return ye3().memoizedState;
  }
  function Xc(e, n2, t) {
    var r = en(e);
    if (t = { lane: r, action: t, hasEagerState: false, eagerState: null, next: null }, Rs(e)) Fs(n2, t);
    else if (t = hs(e, n2, t, r), t !== null) {
      var l3 = b();
      Ce2(t, e, r, l3), Is(t, n2, r);
    }
  }
  function Gc(e, n2, t) {
    var r = en(e), l3 = { lane: r, action: t, hasEagerState: false, eagerState: null, next: null };
    if (Rs(e)) Fs(n2, l3);
    else {
      var i2 = e.alternate;
      if (e.lanes === 0 && (i2 === null || i2.lanes === 0) && (i2 = n2.lastRenderedReducer, i2 !== null)) try {
        var u2 = n2.lastRenderedState, o = i2(u2, t);
        if (l3.hasEagerState = true, l3.eagerState = o, xe3(o, u2)) {
          var s = n2.interleaved;
          s === null ? (l3.next = l3, Ii(n2)) : (l3.next = s.next, s.next = l3), n2.interleaved = l3;
          return;
        }
      } catch {
      } finally {
      }
      t = hs(e, n2, l3, r), t !== null && (l3 = b(), Ce2(t, e, r, l3), Is(t, n2, r));
    }
  }
  function Rs(e) {
    var n2 = e.alternate;
    return e === R2 || n2 !== null && n2 === R2;
  }
  function Fs(e, n2) {
    mt = Mr = true;
    var t = e.pending;
    t === null ? n2.next = n2 : (n2.next = t.next, t.next = n2), e.pending = n2;
  }
  function Is(e, n2, t) {
    if ((t & 4194240) !== 0) {
      var r = n2.lanes;
      r &= e.pendingLanes, t |= r, n2.lanes = t, Ei(e, t);
    }
  }
  var Dr = { readContext: ve3, useCallback: X2, useContext: X2, useEffect: X2, useImperativeHandle: X2, useInsertionEffect: X2, useLayoutEffect: X2, useMemo: X2, useReducer: X2, useRef: X2, useState: X2, useDebugValue: X2, useDeferredValue: X2, useTransition: X2, useMutableSource: X2, useSyncExternalStore: X2, useId: X2, unstable_isNewReconciler: false }, Zc = { readContext: ve3, useCallback: function(e, n2) {
    return _e3().memoizedState = [e, n2 === void 0 ? null : n2], e;
  }, useContext: ve3, useEffect: Ku, useImperativeHandle: function(e, n2, t) {
    return t = t != null ? t.concat([e]) : null, ar(4194308, 4, Ps.bind(null, n2, e), t);
  }, useLayoutEffect: function(e, n2) {
    return ar(4194308, 4, e, n2);
  }, useInsertionEffect: function(e, n2) {
    return ar(4, 2, e, n2);
  }, useMemo: function(e, n2) {
    var t = _e3();
    return n2 = n2 === void 0 ? null : n2, e = e(), t.memoizedState = [e, n2], e;
  }, useReducer: function(e, n2, t) {
    var r = _e3();
    return n2 = t !== void 0 ? t(n2) : n2, r.memoizedState = r.baseState = n2, e = { pending: null, interleaved: null, lanes: 0, dispatch: null, lastRenderedReducer: e, lastRenderedState: n2 }, r.queue = e, e = e.dispatch = Xc.bind(null, R2, e), [r.memoizedState, e];
  }, useRef: function(e) {
    var n2 = _e3();
    return e = { current: e }, n2.memoizedState = e;
  }, useState: $u, useDebugValue: $i, useDeferredValue: function(e) {
    return _e3().memoizedState = e;
  }, useTransition: function() {
    var e = $u(false), n2 = e[0];
    return e = Yc.bind(null, e[1]), _e3().memoizedState = e, [n2, e];
  }, useMutableSource: function() {
  }, useSyncExternalStore: function(e, n2, t) {
    var r = R2, l3 = _e3();
    if (D2) {
      if (t === void 0) throw Error(v2(407));
      t = t();
    } else {
      if (t = n2(), Q === null) throw Error(v2(349));
      (yn & 30) !== 0 || Ss(r, n2, t);
    }
    l3.memoizedState = t;
    var i2 = { value: t, getSnapshot: n2 };
    return l3.queue = i2, Ku(Es.bind(null, r, i2, e), [e]), r.flags |= 2048, Ot(9, ks.bind(null, r, i2, t, n2), void 0, null), t;
  }, useId: function() {
    var e = _e3(), n2 = Q.identifierPrefix;
    if (D2) {
      var t = Oe2, r = De2;
      t = (r & ~(1 << 32 - Ee2(r) - 1)).toString(32) + t, n2 = ":" + n2 + "R" + t, t = Mt++, 0 < t && (n2 += "H" + t.toString(32)), n2 += ":";
    } else t = Kc++, n2 = ":" + n2 + "r" + t.toString(32) + ":";
    return e.memoizedState = n2;
  }, unstable_isNewReconciler: false }, Jc = { readContext: ve3, useCallback: Ts, useContext: ve3, useEffect: Qi, useImperativeHandle: Ls, useInsertionEffect: _s, useLayoutEffect: zs, useMemo: Ms, useReducer: vl, useRef: Ns, useState: function() {
    return vl(Dt);
  }, useDebugValue: $i, useDeferredValue: function(e) {
    var n2 = ye3();
    return Ds(n2, A2.memoizedState, e);
  }, useTransition: function() {
    var e = vl(Dt)[0], n2 = ye3().memoizedState;
    return [e, n2];
  }, useMutableSource: gs, useSyncExternalStore: ws, useId: Os, unstable_isNewReconciler: false }, qc = { readContext: ve3, useCallback: Ts, useContext: ve3, useEffect: Qi, useImperativeHandle: Ls, useInsertionEffect: _s, useLayoutEffect: zs, useMemo: Ms, useReducer: yl, useRef: Ns, useState: function() {
    return yl(Dt);
  }, useDebugValue: $i, useDeferredValue: function(e) {
    var n2 = ye3();
    return A2 === null ? n2.memoizedState = e : Ds(n2, A2.memoizedState, e);
  }, useTransition: function() {
    var e = yl(Dt)[0], n2 = ye3().memoizedState;
    return [e, n2];
  }, useMutableSource: gs, useSyncExternalStore: ws, useId: Os, unstable_isNewReconciler: false };
  function we3(e, n2) {
    if (e && e.defaultProps) {
      n2 = F3({}, n2), e = e.defaultProps;
      for (var t in e) n2[t] === void 0 && (n2[t] = e[t]);
      return n2;
    }
    return n2;
  }
  function ql(e, n2, t, r) {
    n2 = e.memoizedState, t = t(r, n2), t = t == null ? n2 : F3({}, n2, t), e.memoizedState = t, e.lanes === 0 && (e.updateQueue.baseState = t);
  }
  var $r = { isMounted: function(e) {
    return (e = e._reactInternals) ? kn(e) === e : false;
  }, enqueueSetState: function(e, n2, t) {
    e = e._reactInternals;
    var r = b(), l3 = en(e), i2 = Re2(r, l3);
    i2.payload = n2, t != null && (i2.callback = t), n2 = qe2(e, i2, l3), n2 !== null && (Ce2(n2, e, l3, r), or(n2, e, l3));
  }, enqueueReplaceState: function(e, n2, t) {
    e = e._reactInternals;
    var r = b(), l3 = en(e), i2 = Re2(r, l3);
    i2.tag = 1, i2.payload = n2, t != null && (i2.callback = t), n2 = qe2(e, i2, l3), n2 !== null && (Ce2(n2, e, l3, r), or(n2, e, l3));
  }, enqueueForceUpdate: function(e, n2) {
    e = e._reactInternals;
    var t = b(), r = en(e), l3 = Re2(t, r);
    l3.tag = 2, n2 != null && (l3.callback = n2), n2 = qe2(e, l3, r), n2 !== null && (Ce2(n2, e, r, t), or(n2, e, r));
  } };
  function Yu(e, n2, t, r, l3, i2, u2) {
    return e = e.stateNode, typeof e.shouldComponentUpdate == "function" ? e.shouldComponentUpdate(r, i2, u2) : n2.prototype && n2.prototype.isPureReactComponent ? !Nt(t, r) || !Nt(l3, i2) : true;
  }
  function js(e, n2, t) {
    var r = false, l3 = rn, i2 = n2.contextType;
    return typeof i2 == "object" && i2 !== null ? i2 = ve3(i2) : (l3 = le2(n2) ? hn : J.current, r = n2.contextTypes, i2 = (r = r != null) ? Wn(e, l3) : rn), n2 = new n2(t, i2), e.memoizedState = n2.state !== null && n2.state !== void 0 ? n2.state : null, n2.updater = $r, e.stateNode = n2, n2._reactInternals = e, r && (e = e.stateNode, e.__reactInternalMemoizedUnmaskedChildContext = l3, e.__reactInternalMemoizedMaskedChildContext = i2), n2;
  }
  function Xu(e, n2, t, r) {
    e = n2.state, typeof n2.componentWillReceiveProps == "function" && n2.componentWillReceiveProps(t, r), typeof n2.UNSAFE_componentWillReceiveProps == "function" && n2.UNSAFE_componentWillReceiveProps(t, r), n2.state !== e && $r.enqueueReplaceState(n2, n2.state, null);
  }
  function bl(e, n2, t, r) {
    var l3 = e.stateNode;
    l3.props = t, l3.state = e.memoizedState, l3.refs = {}, ji(e);
    var i2 = n2.contextType;
    typeof i2 == "object" && i2 !== null ? l3.context = ve3(i2) : (i2 = le2(n2) ? hn : J.current, l3.context = Wn(e, i2)), l3.state = e.memoizedState, i2 = n2.getDerivedStateFromProps, typeof i2 == "function" && (ql(e, n2, i2, t), l3.state = e.memoizedState), typeof n2.getDerivedStateFromProps == "function" || typeof l3.getSnapshotBeforeUpdate == "function" || typeof l3.UNSAFE_componentWillMount != "function" && typeof l3.componentWillMount != "function" || (n2 = l3.state, typeof l3.componentWillMount == "function" && l3.componentWillMount(), typeof l3.UNSAFE_componentWillMount == "function" && l3.UNSAFE_componentWillMount(), n2 !== l3.state && $r.enqueueReplaceState(l3, l3.state, null), Lr(e, t, l3, r), l3.state = e.memoizedState), typeof l3.componentDidMount == "function" && (e.flags |= 4194308);
  }
  function Yn(e, n2) {
    try {
      var t = "", r = n2;
      do
        t += Pa(r), r = r.return;
      while (r);
      var l3 = t;
    } catch (i2) {
      l3 = `
Error generating stack: ` + i2.message + `
` + i2.stack;
    }
    return { value: e, source: n2, stack: l3, digest: null };
  }
  function gl(e, n2, t) {
    return { value: e, source: null, stack: t ?? null, digest: n2 ?? null };
  }
  function ei(e, n2) {
    try {
      console.error(n2.value);
    } catch (t) {
      setTimeout(function() {
        throw t;
      });
    }
  }
  var bc = typeof WeakMap == "function" ? WeakMap : Map;
  function Us(e, n2, t) {
    t = Re2(-1, t), t.tag = 3, t.payload = { element: null };
    var r = n2.value;
    return t.callback = function() {
      Rr || (Rr = true, ci = r), ei(e, n2);
    }, t;
  }
  function Vs(e, n2, t) {
    t = Re2(-1, t), t.tag = 3;
    var r = e.type.getDerivedStateFromError;
    if (typeof r == "function") {
      var l3 = n2.value;
      t.payload = function() {
        return r(l3);
      }, t.callback = function() {
        ei(e, n2);
      };
    }
    var i2 = e.stateNode;
    return i2 !== null && typeof i2.componentDidCatch == "function" && (t.callback = function() {
      ei(e, n2), typeof r != "function" && (be3 === null ? be3 = /* @__PURE__ */ new Set([this]) : be3.add(this));
      var u2 = n2.stack;
      this.componentDidCatch(n2.value, { componentStack: u2 !== null ? u2 : "" });
    }), t;
  }
  function Gu(e, n2, t) {
    var r = e.pingCache;
    if (r === null) {
      r = e.pingCache = new bc();
      var l3 = /* @__PURE__ */ new Set();
      r.set(n2, l3);
    } else l3 = r.get(n2), l3 === void 0 && (l3 = /* @__PURE__ */ new Set(), r.set(n2, l3));
    l3.has(t) || (l3.add(t), e = mf.bind(null, e, n2, t), n2.then(e, e));
  }
  function Zu(e) {
    do {
      var n2;
      if ((n2 = e.tag === 13) && (n2 = e.memoizedState, n2 = n2 !== null ? n2.dehydrated !== null : true), n2) return e;
      e = e.return;
    } while (e !== null);
    return null;
  }
  function Ju(e, n2, t, r, l3) {
    return (e.mode & 1) === 0 ? (e === n2 ? e.flags |= 65536 : (e.flags |= 128, t.flags |= 131072, t.flags &= -52805, t.tag === 1 && (t.alternate === null ? t.tag = 17 : (n2 = Re2(-1, 1), n2.tag = 2, qe2(t, n2, 1))), t.lanes |= 1), e) : (e.flags |= 65536, e.lanes = l3, e);
  }
  var ef = Ve2.ReactCurrentOwner, te = false;
  function q2(e, n2, t, r) {
    n2.child = e === null ? ms(n2, null, t, r) : $n(n2, e.child, t, r);
  }
  function qu(e, n2, t, r, l3) {
    t = t.render;
    var i2 = n2.ref;
    return An(n2, l3), r = Hi(e, n2, t, r, i2, l3), t = Wi(), e !== null && !te ? (n2.updateQueue = e.updateQueue, n2.flags &= -2053, e.lanes &= ~l3, Ue2(e, n2, l3)) : (D2 && t && Ti(n2), n2.flags |= 1, q2(e, n2, r, l3), n2.child);
  }
  function bu(e, n2, t, r, l3) {
    if (e === null) {
      var i2 = t.type;
      return typeof i2 == "function" && !bi(i2) && i2.defaultProps === void 0 && t.compare === null && t.defaultProps === void 0 ? (n2.tag = 15, n2.type = i2, As(e, n2, i2, r, l3)) : (e = pr(t.type, null, r, n2, n2.mode, l3), e.ref = n2.ref, e.return = n2, n2.child = e);
    }
    if (i2 = e.child, (e.lanes & l3) === 0) {
      var u2 = i2.memoizedProps;
      if (t = t.compare, t = t !== null ? t : Nt, t(u2, r) && e.ref === n2.ref) return Ue2(e, n2, l3);
    }
    return n2.flags |= 1, e = nn(i2, r), e.ref = n2.ref, e.return = n2, n2.child = e;
  }
  function As(e, n2, t, r, l3) {
    if (e !== null) {
      var i2 = e.memoizedProps;
      if (Nt(i2, r) && e.ref === n2.ref) if (te = false, n2.pendingProps = r = i2, (e.lanes & l3) !== 0) (e.flags & 131072) !== 0 && (te = true);
      else return n2.lanes = e.lanes, Ue2(e, n2, l3);
    }
    return ni(e, n2, t, r, l3);
  }
  function Bs(e, n2, t) {
    var r = n2.pendingProps, l3 = r.children, i2 = e !== null ? e.memoizedState : null;
    if (r.mode === "hidden") if ((n2.mode & 1) === 0) n2.memoizedState = { baseLanes: 0, cachePool: null, transitions: null }, L3(Fn, ue2), ue2 |= t;
    else {
      if ((t & 1073741824) === 0) return e = i2 !== null ? i2.baseLanes | t : t, n2.lanes = n2.childLanes = 1073741824, n2.memoizedState = { baseLanes: e, cachePool: null, transitions: null }, n2.updateQueue = null, L3(Fn, ue2), ue2 |= e, null;
      n2.memoizedState = { baseLanes: 0, cachePool: null, transitions: null }, r = i2 !== null ? i2.baseLanes : t, L3(Fn, ue2), ue2 |= r;
    }
    else i2 !== null ? (r = i2.baseLanes | t, n2.memoizedState = null) : r = t, L3(Fn, ue2), ue2 |= r;
    return q2(e, n2, l3, t), n2.child;
  }
  function Hs(e, n2) {
    var t = n2.ref;
    (e === null && t !== null || e !== null && e.ref !== t) && (n2.flags |= 512, n2.flags |= 2097152);
  }
  function ni(e, n2, t, r, l3) {
    var i2 = le2(t) ? hn : J.current;
    return i2 = Wn(n2, i2), An(n2, l3), t = Hi(e, n2, t, r, i2, l3), r = Wi(), e !== null && !te ? (n2.updateQueue = e.updateQueue, n2.flags &= -2053, e.lanes &= ~l3, Ue2(e, n2, l3)) : (D2 && r && Ti(n2), n2.flags |= 1, q2(e, n2, t, l3), n2.child);
  }
  function eo(e, n2, t, r, l3) {
    if (le2(t)) {
      var i2 = true;
      xr(n2);
    } else i2 = false;
    if (An(n2, l3), n2.stateNode === null) cr(e, n2), js(n2, t, r), bl(n2, t, r, l3), r = true;
    else if (e === null) {
      var u2 = n2.stateNode, o = n2.memoizedProps;
      u2.props = o;
      var s = u2.context, d3 = t.contextType;
      typeof d3 == "object" && d3 !== null ? d3 = ve3(d3) : (d3 = le2(t) ? hn : J.current, d3 = Wn(n2, d3));
      var m2 = t.getDerivedStateFromProps, h3 = typeof m2 == "function" || typeof u2.getSnapshotBeforeUpdate == "function";
      h3 || typeof u2.UNSAFE_componentWillReceiveProps != "function" && typeof u2.componentWillReceiveProps != "function" || (o !== r || s !== d3) && Xu(n2, u2, r, d3), We2 = false;
      var p = n2.memoizedState;
      u2.state = p, Lr(n2, r, u2, l3), s = n2.memoizedState, o !== r || p !== s || re.current || We2 ? (typeof m2 == "function" && (ql(n2, t, m2, r), s = n2.memoizedState), (o = We2 || Yu(n2, t, o, r, p, s, d3)) ? (h3 || typeof u2.UNSAFE_componentWillMount != "function" && typeof u2.componentWillMount != "function" || (typeof u2.componentWillMount == "function" && u2.componentWillMount(), typeof u2.UNSAFE_componentWillMount == "function" && u2.UNSAFE_componentWillMount()), typeof u2.componentDidMount == "function" && (n2.flags |= 4194308)) : (typeof u2.componentDidMount == "function" && (n2.flags |= 4194308), n2.memoizedProps = r, n2.memoizedState = s), u2.props = r, u2.state = s, u2.context = d3, r = o) : (typeof u2.componentDidMount == "function" && (n2.flags |= 4194308), r = false);
    } else {
      u2 = n2.stateNode, vs(e, n2), o = n2.memoizedProps, d3 = n2.type === n2.elementType ? o : we3(n2.type, o), u2.props = d3, h3 = n2.pendingProps, p = u2.context, s = t.contextType, typeof s == "object" && s !== null ? s = ve3(s) : (s = le2(t) ? hn : J.current, s = Wn(n2, s));
      var g2 = t.getDerivedStateFromProps;
      (m2 = typeof g2 == "function" || typeof u2.getSnapshotBeforeUpdate == "function") || typeof u2.UNSAFE_componentWillReceiveProps != "function" && typeof u2.componentWillReceiveProps != "function" || (o !== h3 || p !== s) && Xu(n2, u2, r, s), We2 = false, p = n2.memoizedState, u2.state = p, Lr(n2, r, u2, l3);
      var S2 = n2.memoizedState;
      o !== h3 || p !== S2 || re.current || We2 ? (typeof g2 == "function" && (ql(n2, t, g2, r), S2 = n2.memoizedState), (d3 = We2 || Yu(n2, t, d3, r, p, S2, s) || false) ? (m2 || typeof u2.UNSAFE_componentWillUpdate != "function" && typeof u2.componentWillUpdate != "function" || (typeof u2.componentWillUpdate == "function" && u2.componentWillUpdate(r, S2, s), typeof u2.UNSAFE_componentWillUpdate == "function" && u2.UNSAFE_componentWillUpdate(r, S2, s)), typeof u2.componentDidUpdate == "function" && (n2.flags |= 4), typeof u2.getSnapshotBeforeUpdate == "function" && (n2.flags |= 1024)) : (typeof u2.componentDidUpdate != "function" || o === e.memoizedProps && p === e.memoizedState || (n2.flags |= 4), typeof u2.getSnapshotBeforeUpdate != "function" || o === e.memoizedProps && p === e.memoizedState || (n2.flags |= 1024), n2.memoizedProps = r, n2.memoizedState = S2), u2.props = r, u2.state = S2, u2.context = s, r = d3) : (typeof u2.componentDidUpdate != "function" || o === e.memoizedProps && p === e.memoizedState || (n2.flags |= 4), typeof u2.getSnapshotBeforeUpdate != "function" || o === e.memoizedProps && p === e.memoizedState || (n2.flags |= 1024), r = false);
    }
    return ti(e, n2, t, r, i2, l3);
  }
  function ti(e, n2, t, r, l3, i2) {
    Hs(e, n2);
    var u2 = (n2.flags & 128) !== 0;
    if (!r && !u2) return l3 && Vu(n2, t, false), Ue2(e, n2, i2);
    r = n2.stateNode, ef.current = n2;
    var o = u2 && typeof t.getDerivedStateFromError != "function" ? null : r.render();
    return n2.flags |= 1, e !== null && u2 ? (n2.child = $n(n2, e.child, null, i2), n2.child = $n(n2, null, o, i2)) : q2(e, n2, o, i2), n2.memoizedState = r.state, l3 && Vu(n2, t, true), n2.child;
  }
  function Ws(e) {
    var n2 = e.stateNode;
    n2.pendingContext ? Uu(e, n2.pendingContext, n2.pendingContext !== n2.context) : n2.context && Uu(e, n2.context, false), Ui(e, n2.containerInfo);
  }
  function no(e, n2, t, r, l3) {
    return Qn(), Di(l3), n2.flags |= 256, q2(e, n2, t, r), n2.child;
  }
  var ri = { dehydrated: null, treeContext: null, retryLane: 0 };
  function li(e) {
    return { baseLanes: e, cachePool: null, transitions: null };
  }
  function Qs(e, n2, t) {
    var r = n2.pendingProps, l3 = O3.current, i2 = false, u2 = (n2.flags & 128) !== 0, o;
    if ((o = u2) || (o = e !== null && e.memoizedState === null ? false : (l3 & 2) !== 0), o ? (i2 = true, n2.flags &= -129) : (e === null || e.memoizedState !== null) && (l3 |= 1), L3(O3, l3 & 1), e === null) return Zl(n2), e = n2.memoizedState, e !== null && (e = e.dehydrated, e !== null) ? ((n2.mode & 1) === 0 ? n2.lanes = 1 : e.data === "$!" ? n2.lanes = 8 : n2.lanes = 1073741824, null) : (u2 = r.children, e = r.fallback, i2 ? (r = n2.mode, i2 = n2.child, u2 = { mode: "hidden", children: u2 }, (r & 1) === 0 && i2 !== null ? (i2.childLanes = 0, i2.pendingProps = u2) : i2 = Xr(u2, r, 0, null), e = mn(e, r, t, null), i2.return = n2, e.return = n2, i2.sibling = e, n2.child = i2, n2.child.memoizedState = li(t), n2.memoizedState = ri, e) : Ki(n2, u2));
    if (l3 = e.memoizedState, l3 !== null && (o = l3.dehydrated, o !== null)) return nf(e, n2, u2, r, o, l3, t);
    if (i2) {
      i2 = r.fallback, u2 = n2.mode, l3 = e.child, o = l3.sibling;
      var s = { mode: "hidden", children: r.children };
      return (u2 & 1) === 0 && n2.child !== l3 ? (r = n2.child, r.childLanes = 0, r.pendingProps = s, n2.deletions = null) : (r = nn(l3, s), r.subtreeFlags = l3.subtreeFlags & 14680064), o !== null ? i2 = nn(o, i2) : (i2 = mn(i2, u2, t, null), i2.flags |= 2), i2.return = n2, r.return = n2, r.sibling = i2, n2.child = r, r = i2, i2 = n2.child, u2 = e.child.memoizedState, u2 = u2 === null ? li(t) : { baseLanes: u2.baseLanes | t, cachePool: null, transitions: u2.transitions }, i2.memoizedState = u2, i2.childLanes = e.childLanes & ~t, n2.memoizedState = ri, r;
    }
    return i2 = e.child, e = i2.sibling, r = nn(i2, { mode: "visible", children: r.children }), (n2.mode & 1) === 0 && (r.lanes = t), r.return = n2, r.sibling = null, e !== null && (t = n2.deletions, t === null ? (n2.deletions = [e], n2.flags |= 16) : t.push(e)), n2.child = r, n2.memoizedState = null, r;
  }
  function Ki(e, n2) {
    return n2 = Xr({ mode: "visible", children: n2 }, e.mode, 0, null), n2.return = e, e.child = n2;
  }
  function nr(e, n2, t, r) {
    return r !== null && Di(r), $n(n2, e.child, null, t), e = Ki(n2, n2.pendingProps.children), e.flags |= 2, n2.memoizedState = null, e;
  }
  function nf(e, n2, t, r, l3, i2, u2) {
    if (t) return n2.flags & 256 ? (n2.flags &= -257, r = gl(Error(v2(422))), nr(e, n2, u2, r)) : n2.memoizedState !== null ? (n2.child = e.child, n2.flags |= 128, null) : (i2 = r.fallback, l3 = n2.mode, r = Xr({ mode: "visible", children: r.children }, l3, 0, null), i2 = mn(i2, l3, u2, null), i2.flags |= 2, r.return = n2, i2.return = n2, r.sibling = i2, n2.child = r, (n2.mode & 1) !== 0 && $n(n2, e.child, null, u2), n2.child.memoizedState = li(u2), n2.memoizedState = ri, i2);
    if ((n2.mode & 1) === 0) return nr(e, n2, u2, null);
    if (l3.data === "$!") {
      if (r = l3.nextSibling && l3.nextSibling.dataset, r) var o = r.dgst;
      return r = o, i2 = Error(v2(419)), r = gl(i2, r, void 0), nr(e, n2, u2, r);
    }
    if (o = (u2 & e.childLanes) !== 0, te || o) {
      if (r = Q, r !== null) {
        switch (u2 & -u2) {
          case 4:
            l3 = 2;
            break;
          case 16:
            l3 = 8;
            break;
          case 64:
          case 128:
          case 256:
          case 512:
          case 1024:
          case 2048:
          case 4096:
          case 8192:
          case 16384:
          case 32768:
          case 65536:
          case 131072:
          case 262144:
          case 524288:
          case 1048576:
          case 2097152:
          case 4194304:
          case 8388608:
          case 16777216:
          case 33554432:
          case 67108864:
            l3 = 32;
            break;
          case 536870912:
            l3 = 268435456;
            break;
          default:
            l3 = 0;
        }
        l3 = (l3 & (r.suspendedLanes | u2)) !== 0 ? 0 : l3, l3 !== 0 && l3 !== i2.retryLane && (i2.retryLane = l3, je2(e, l3), Ce2(r, e, l3, -1));
      }
      return qi(), r = gl(Error(v2(421))), nr(e, n2, u2, r);
    }
    return l3.data === "$?" ? (n2.flags |= 128, n2.child = e.child, n2 = hf.bind(null, e), l3._reactRetry = n2, null) : (e = i2.treeContext, oe2 = Je(l3.nextSibling), se2 = n2, D2 = true, ke3 = null, e !== null && (de3[pe3++] = De2, de3[pe3++] = Oe2, de3[pe3++] = vn, De2 = e.id, Oe2 = e.overflow, vn = n2), n2 = Ki(n2, r.children), n2.flags |= 4096, n2);
  }
  function to(e, n2, t) {
    e.lanes |= n2;
    var r = e.alternate;
    r !== null && (r.lanes |= n2), Jl(e.return, n2, t);
  }
  function wl(e, n2, t, r, l3) {
    var i2 = e.memoizedState;
    i2 === null ? e.memoizedState = { isBackwards: n2, rendering: null, renderingStartTime: 0, last: r, tail: t, tailMode: l3 } : (i2.isBackwards = n2, i2.rendering = null, i2.renderingStartTime = 0, i2.last = r, i2.tail = t, i2.tailMode = l3);
  }
  function $s(e, n2, t) {
    var r = n2.pendingProps, l3 = r.revealOrder, i2 = r.tail;
    if (q2(e, n2, r.children, t), r = O3.current, (r & 2) !== 0) r = r & 1 | 2, n2.flags |= 128;
    else {
      if (e !== null && (e.flags & 128) !== 0) e: for (e = n2.child; e !== null; ) {
        if (e.tag === 13) e.memoizedState !== null && to(e, t, n2);
        else if (e.tag === 19) to(e, t, n2);
        else if (e.child !== null) {
          e.child.return = e, e = e.child;
          continue;
        }
        if (e === n2) break e;
        for (; e.sibling === null; ) {
          if (e.return === null || e.return === n2) break e;
          e = e.return;
        }
        e.sibling.return = e.return, e = e.sibling;
      }
      r &= 1;
    }
    if (L3(O3, r), (n2.mode & 1) === 0) n2.memoizedState = null;
    else switch (l3) {
      case "forwards":
        for (t = n2.child, l3 = null; t !== null; ) e = t.alternate, e !== null && Tr(e) === null && (l3 = t), t = t.sibling;
        t = l3, t === null ? (l3 = n2.child, n2.child = null) : (l3 = t.sibling, t.sibling = null), wl(n2, false, l3, t, i2);
        break;
      case "backwards":
        for (t = null, l3 = n2.child, n2.child = null; l3 !== null; ) {
          if (e = l3.alternate, e !== null && Tr(e) === null) {
            n2.child = l3;
            break;
          }
          e = l3.sibling, l3.sibling = t, t = l3, l3 = e;
        }
        wl(n2, true, t, null, i2);
        break;
      case "together":
        wl(n2, false, null, null, void 0);
        break;
      default:
        n2.memoizedState = null;
    }
    return n2.child;
  }
  function cr(e, n2) {
    (n2.mode & 1) === 0 && e !== null && (e.alternate = null, n2.alternate = null, n2.flags |= 2);
  }
  function Ue2(e, n2, t) {
    if (e !== null && (n2.dependencies = e.dependencies), gn |= n2.lanes, (t & n2.childLanes) === 0) return null;
    if (e !== null && n2.child !== e.child) throw Error(v2(153));
    if (n2.child !== null) {
      for (e = n2.child, t = nn(e, e.pendingProps), n2.child = t, t.return = n2; e.sibling !== null; ) e = e.sibling, t = t.sibling = nn(e, e.pendingProps), t.return = n2;
      t.sibling = null;
    }
    return n2.child;
  }
  function tf(e, n2, t) {
    switch (n2.tag) {
      case 3:
        Ws(n2), Qn();
        break;
      case 5:
        ys(n2);
        break;
      case 1:
        le2(n2.type) && xr(n2);
        break;
      case 4:
        Ui(n2, n2.stateNode.containerInfo);
        break;
      case 10:
        var r = n2.type._context, l3 = n2.memoizedProps.value;
        L3(zr, r._currentValue), r._currentValue = l3;
        break;
      case 13:
        if (r = n2.memoizedState, r !== null) return r.dehydrated !== null ? (L3(O3, O3.current & 1), n2.flags |= 128, null) : (t & n2.child.childLanes) !== 0 ? Qs(e, n2, t) : (L3(O3, O3.current & 1), e = Ue2(e, n2, t), e !== null ? e.sibling : null);
        L3(O3, O3.current & 1);
        break;
      case 19:
        if (r = (t & n2.childLanes) !== 0, (e.flags & 128) !== 0) {
          if (r) return $s(e, n2, t);
          n2.flags |= 128;
        }
        if (l3 = n2.memoizedState, l3 !== null && (l3.rendering = null, l3.tail = null, l3.lastEffect = null), L3(O3, O3.current), r) break;
        return null;
      case 22:
      case 23:
        return n2.lanes = 0, Bs(e, n2, t);
    }
    return Ue2(e, n2, t);
  }
  var Ks, ii, Ys, Xs;
  Ks = function(e, n2) {
    for (var t = n2.child; t !== null; ) {
      if (t.tag === 5 || t.tag === 6) e.appendChild(t.stateNode);
      else if (t.tag !== 4 && t.child !== null) {
        t.child.return = t, t = t.child;
        continue;
      }
      if (t === n2) break;
      for (; t.sibling === null; ) {
        if (t.return === null || t.return === n2) return;
        t = t.return;
      }
      t.sibling.return = t.return, t = t.sibling;
    }
  };
  ii = function() {
  };
  Ys = function(e, n2, t, r) {
    var l3 = e.memoizedProps;
    if (l3 !== r) {
      e = n2.stateNode, dn(Le2.current);
      var i2 = null;
      switch (t) {
        case "input":
          l3 = zl(e, l3), r = zl(e, r), i2 = [];
          break;
        case "select":
          l3 = F3({}, l3, { value: void 0 }), r = F3({}, r, { value: void 0 }), i2 = [];
          break;
        case "textarea":
          l3 = Tl(e, l3), r = Tl(e, r), i2 = [];
          break;
        default:
          typeof l3.onClick != "function" && typeof r.onClick == "function" && (e.onclick = Er);
      }
      Dl(t, r);
      var u2;
      t = null;
      for (d3 in l3) if (!r.hasOwnProperty(d3) && l3.hasOwnProperty(d3) && l3[d3] != null) if (d3 === "style") {
        var o = l3[d3];
        for (u2 in o) o.hasOwnProperty(u2) && (t || (t = {}), t[u2] = "");
      } else d3 !== "dangerouslySetInnerHTML" && d3 !== "children" && d3 !== "suppressContentEditableWarning" && d3 !== "suppressHydrationWarning" && d3 !== "autoFocus" && (gt.hasOwnProperty(d3) ? i2 || (i2 = []) : (i2 = i2 || []).push(d3, null));
      for (d3 in r) {
        var s = r[d3];
        if (o = l3?.[d3], r.hasOwnProperty(d3) && s !== o && (s != null || o != null)) if (d3 === "style") if (o) {
          for (u2 in o) !o.hasOwnProperty(u2) || s && s.hasOwnProperty(u2) || (t || (t = {}), t[u2] = "");
          for (u2 in s) s.hasOwnProperty(u2) && o[u2] !== s[u2] && (t || (t = {}), t[u2] = s[u2]);
        } else t || (i2 || (i2 = []), i2.push(d3, t)), t = s;
        else d3 === "dangerouslySetInnerHTML" ? (s = s ? s.__html : void 0, o = o ? o.__html : void 0, s != null && o !== s && (i2 = i2 || []).push(d3, s)) : d3 === "children" ? typeof s != "string" && typeof s != "number" || (i2 = i2 || []).push(d3, "" + s) : d3 !== "suppressContentEditableWarning" && d3 !== "suppressHydrationWarning" && (gt.hasOwnProperty(d3) ? (s != null && d3 === "onScroll" && T2("scroll", e), i2 || o === s || (i2 = [])) : (i2 = i2 || []).push(d3, s));
      }
      t && (i2 = i2 || []).push("style", t);
      var d3 = i2;
      (n2.updateQueue = d3) && (n2.flags |= 4);
    }
  };
  Xs = function(e, n2, t, r) {
    t !== r && (n2.flags |= 4);
  };
  function tt(e, n2) {
    if (!D2) switch (e.tailMode) {
      case "hidden":
        n2 = e.tail;
        for (var t = null; n2 !== null; ) n2.alternate !== null && (t = n2), n2 = n2.sibling;
        t === null ? e.tail = null : t.sibling = null;
        break;
      case "collapsed":
        t = e.tail;
        for (var r = null; t !== null; ) t.alternate !== null && (r = t), t = t.sibling;
        r === null ? n2 || e.tail === null ? e.tail = null : e.tail.sibling = null : r.sibling = null;
    }
  }
  function G(e) {
    var n2 = e.alternate !== null && e.alternate.child === e.child, t = 0, r = 0;
    if (n2) for (var l3 = e.child; l3 !== null; ) t |= l3.lanes | l3.childLanes, r |= l3.subtreeFlags & 14680064, r |= l3.flags & 14680064, l3.return = e, l3 = l3.sibling;
    else for (l3 = e.child; l3 !== null; ) t |= l3.lanes | l3.childLanes, r |= l3.subtreeFlags, r |= l3.flags, l3.return = e, l3 = l3.sibling;
    return e.subtreeFlags |= r, e.childLanes = t, n2;
  }
  function rf(e, n2, t) {
    var r = n2.pendingProps;
    switch (Mi(n2), n2.tag) {
      case 2:
      case 16:
      case 15:
      case 0:
      case 11:
      case 7:
      case 8:
      case 12:
      case 9:
      case 14:
        return G(n2), null;
      case 1:
        return le2(n2.type) && Cr(), G(n2), null;
      case 3:
        return r = n2.stateNode, Kn(), M2(re), M2(J), Ai(), r.pendingContext && (r.context = r.pendingContext, r.pendingContext = null), (e === null || e.child === null) && (bt(n2) ? n2.flags |= 4 : e === null || e.memoizedState.isDehydrated && (n2.flags & 256) === 0 || (n2.flags |= 1024, ke3 !== null && (pi(ke3), ke3 = null))), ii(e, n2), G(n2), null;
      case 5:
        Vi(n2);
        var l3 = dn(Tt.current);
        if (t = n2.type, e !== null && n2.stateNode != null) Ys(e, n2, t, r, l3), e.ref !== n2.ref && (n2.flags |= 512, n2.flags |= 2097152);
        else {
          if (!r) {
            if (n2.stateNode === null) throw Error(v2(166));
            return G(n2), null;
          }
          if (e = dn(Le2.current), bt(n2)) {
            r = n2.stateNode, t = n2.type;
            var i2 = n2.memoizedProps;
            switch (r[ze2] = n2, r[Pt] = i2, e = (n2.mode & 1) !== 0, t) {
              case "dialog":
                T2("cancel", r), T2("close", r);
                break;
              case "iframe":
              case "object":
              case "embed":
                T2("load", r);
                break;
              case "video":
              case "audio":
                for (l3 = 0; l3 < st.length; l3++) T2(st[l3], r);
                break;
              case "source":
                T2("error", r);
                break;
              case "img":
              case "image":
              case "link":
                T2("error", r), T2("load", r);
                break;
              case "details":
                T2("toggle", r);
                break;
              case "input":
                cu(r, i2), T2("invalid", r);
                break;
              case "select":
                r._wrapperState = { wasMultiple: !!i2.multiple }, T2("invalid", r);
                break;
              case "textarea":
                du(r, i2), T2("invalid", r);
            }
            Dl(t, i2), l3 = null;
            for (var u2 in i2) if (i2.hasOwnProperty(u2)) {
              var o = i2[u2];
              u2 === "children" ? typeof o == "string" ? r.textContent !== o && (i2.suppressHydrationWarning !== true && qt(r.textContent, o, e), l3 = ["children", o]) : typeof o == "number" && r.textContent !== "" + o && (i2.suppressHydrationWarning !== true && qt(r.textContent, o, e), l3 = ["children", "" + o]) : gt.hasOwnProperty(u2) && o != null && u2 === "onScroll" && T2("scroll", r);
            }
            switch (t) {
              case "input":
                At(r), fu(r, i2, true);
                break;
              case "textarea":
                At(r), pu(r);
                break;
              case "select":
              case "option":
                break;
              default:
                typeof i2.onClick == "function" && (r.onclick = Er);
            }
            r = l3, n2.updateQueue = r, r !== null && (n2.flags |= 4);
          } else {
            u2 = l3.nodeType === 9 ? l3 : l3.ownerDocument, e === "http://www.w3.org/1999/xhtml" && (e = Eo(t)), e === "http://www.w3.org/1999/xhtml" ? t === "script" ? (e = u2.createElement("div"), e.innerHTML = "<script><\/script>", e = e.removeChild(e.firstChild)) : typeof r.is == "string" ? e = u2.createElement(t, { is: r.is }) : (e = u2.createElement(t), t === "select" && (u2 = e, r.multiple ? u2.multiple = true : r.size && (u2.size = r.size))) : e = u2.createElementNS(e, t), e[ze2] = n2, e[Pt] = r, Ks(e, n2, false, false), n2.stateNode = e;
            e: {
              switch (u2 = Ol(t, r), t) {
                case "dialog":
                  T2("cancel", e), T2("close", e), l3 = r;
                  break;
                case "iframe":
                case "object":
                case "embed":
                  T2("load", e), l3 = r;
                  break;
                case "video":
                case "audio":
                  for (l3 = 0; l3 < st.length; l3++) T2(st[l3], e);
                  l3 = r;
                  break;
                case "source":
                  T2("error", e), l3 = r;
                  break;
                case "img":
                case "image":
                case "link":
                  T2("error", e), T2("load", e), l3 = r;
                  break;
                case "details":
                  T2("toggle", e), l3 = r;
                  break;
                case "input":
                  cu(e, r), l3 = zl(e, r), T2("invalid", e);
                  break;
                case "option":
                  l3 = r;
                  break;
                case "select":
                  e._wrapperState = { wasMultiple: !!r.multiple }, l3 = F3({}, r, { value: void 0 }), T2("invalid", e);
                  break;
                case "textarea":
                  du(e, r), l3 = Tl(e, r), T2("invalid", e);
                  break;
                default:
                  l3 = r;
              }
              Dl(t, l3), o = l3;
              for (i2 in o) if (o.hasOwnProperty(i2)) {
                var s = o[i2];
                i2 === "style" ? No(e, s) : i2 === "dangerouslySetInnerHTML" ? (s = s ? s.__html : void 0, s != null && Co(e, s)) : i2 === "children" ? typeof s == "string" ? (t !== "textarea" || s !== "") && wt(e, s) : typeof s == "number" && wt(e, "" + s) : i2 !== "suppressContentEditableWarning" && i2 !== "suppressHydrationWarning" && i2 !== "autoFocus" && (gt.hasOwnProperty(i2) ? s != null && i2 === "onScroll" && T2("scroll", e) : s != null && vi(e, i2, s, u2));
              }
              switch (t) {
                case "input":
                  At(e), fu(e, r, false);
                  break;
                case "textarea":
                  At(e), pu(e);
                  break;
                case "option":
                  r.value != null && e.setAttribute("value", "" + tn(r.value));
                  break;
                case "select":
                  e.multiple = !!r.multiple, i2 = r.value, i2 != null ? In(e, !!r.multiple, i2, false) : r.defaultValue != null && In(e, !!r.multiple, r.defaultValue, true);
                  break;
                default:
                  typeof l3.onClick == "function" && (e.onclick = Er);
              }
              switch (t) {
                case "button":
                case "input":
                case "select":
                case "textarea":
                  r = !!r.autoFocus;
                  break e;
                case "img":
                  r = true;
                  break e;
                default:
                  r = false;
              }
            }
            r && (n2.flags |= 4);
          }
          n2.ref !== null && (n2.flags |= 512, n2.flags |= 2097152);
        }
        return G(n2), null;
      case 6:
        if (e && n2.stateNode != null) Xs(e, n2, e.memoizedProps, r);
        else {
          if (typeof r != "string" && n2.stateNode === null) throw Error(v2(166));
          if (t = dn(Tt.current), dn(Le2.current), bt(n2)) {
            if (r = n2.stateNode, t = n2.memoizedProps, r[ze2] = n2, (i2 = r.nodeValue !== t) && (e = se2, e !== null)) switch (e.tag) {
              case 3:
                qt(r.nodeValue, t, (e.mode & 1) !== 0);
                break;
              case 5:
                e.memoizedProps.suppressHydrationWarning !== true && qt(r.nodeValue, t, (e.mode & 1) !== 0);
            }
            i2 && (n2.flags |= 4);
          } else r = (t.nodeType === 9 ? t : t.ownerDocument).createTextNode(r), r[ze2] = n2, n2.stateNode = r;
        }
        return G(n2), null;
      case 13:
        if (M2(O3), r = n2.memoizedState, e === null || e.memoizedState !== null && e.memoizedState.dehydrated !== null) {
          if (D2 && oe2 !== null && (n2.mode & 1) !== 0 && (n2.flags & 128) === 0) ds(), Qn(), n2.flags |= 98560, i2 = false;
          else if (i2 = bt(n2), r !== null && r.dehydrated !== null) {
            if (e === null) {
              if (!i2) throw Error(v2(318));
              if (i2 = n2.memoizedState, i2 = i2 !== null ? i2.dehydrated : null, !i2) throw Error(v2(317));
              i2[ze2] = n2;
            } else Qn(), (n2.flags & 128) === 0 && (n2.memoizedState = null), n2.flags |= 4;
            G(n2), i2 = false;
          } else ke3 !== null && (pi(ke3), ke3 = null), i2 = true;
          if (!i2) return n2.flags & 65536 ? n2 : null;
        }
        return (n2.flags & 128) !== 0 ? (n2.lanes = t, n2) : (r = r !== null, r !== (e !== null && e.memoizedState !== null) && r && (n2.child.flags |= 8192, (n2.mode & 1) !== 0 && (e === null || (O3.current & 1) !== 0 ? B3 === 0 && (B3 = 3) : qi())), n2.updateQueue !== null && (n2.flags |= 4), G(n2), null);
      case 4:
        return Kn(), ii(e, n2), e === null && _t(n2.stateNode.containerInfo), G(n2), null;
      case 10:
        return Fi(n2.type._context), G(n2), null;
      case 17:
        return le2(n2.type) && Cr(), G(n2), null;
      case 19:
        if (M2(O3), i2 = n2.memoizedState, i2 === null) return G(n2), null;
        if (r = (n2.flags & 128) !== 0, u2 = i2.rendering, u2 === null) if (r) tt(i2, false);
        else {
          if (B3 !== 0 || e !== null && (e.flags & 128) !== 0) for (e = n2.child; e !== null; ) {
            if (u2 = Tr(e), u2 !== null) {
              for (n2.flags |= 128, tt(i2, false), r = u2.updateQueue, r !== null && (n2.updateQueue = r, n2.flags |= 4), n2.subtreeFlags = 0, r = t, t = n2.child; t !== null; ) i2 = t, e = r, i2.flags &= 14680066, u2 = i2.alternate, u2 === null ? (i2.childLanes = 0, i2.lanes = e, i2.child = null, i2.subtreeFlags = 0, i2.memoizedProps = null, i2.memoizedState = null, i2.updateQueue = null, i2.dependencies = null, i2.stateNode = null) : (i2.childLanes = u2.childLanes, i2.lanes = u2.lanes, i2.child = u2.child, i2.subtreeFlags = 0, i2.deletions = null, i2.memoizedProps = u2.memoizedProps, i2.memoizedState = u2.memoizedState, i2.updateQueue = u2.updateQueue, i2.type = u2.type, e = u2.dependencies, i2.dependencies = e === null ? null : { lanes: e.lanes, firstContext: e.firstContext }), t = t.sibling;
              return L3(O3, O3.current & 1 | 2), n2.child;
            }
            e = e.sibling;
          }
          i2.tail !== null && U3() > Xn && (n2.flags |= 128, r = true, tt(i2, false), n2.lanes = 4194304);
        }
        else {
          if (!r) if (e = Tr(u2), e !== null) {
            if (n2.flags |= 128, r = true, t = e.updateQueue, t !== null && (n2.updateQueue = t, n2.flags |= 4), tt(i2, true), i2.tail === null && i2.tailMode === "hidden" && !u2.alternate && !D2) return G(n2), null;
          } else 2 * U3() - i2.renderingStartTime > Xn && t !== 1073741824 && (n2.flags |= 128, r = true, tt(i2, false), n2.lanes = 4194304);
          i2.isBackwards ? (u2.sibling = n2.child, n2.child = u2) : (t = i2.last, t !== null ? t.sibling = u2 : n2.child = u2, i2.last = u2);
        }
        return i2.tail !== null ? (n2 = i2.tail, i2.rendering = n2, i2.tail = n2.sibling, i2.renderingStartTime = U3(), n2.sibling = null, t = O3.current, L3(O3, r ? t & 1 | 2 : t & 1), n2) : (G(n2), null);
      case 22:
      case 23:
        return Ji(), r = n2.memoizedState !== null, e !== null && e.memoizedState !== null !== r && (n2.flags |= 8192), r && (n2.mode & 1) !== 0 ? (ue2 & 1073741824) !== 0 && (G(n2), n2.subtreeFlags & 6 && (n2.flags |= 8192)) : G(n2), null;
      case 24:
        return null;
      case 25:
        return null;
    }
    throw Error(v2(156, n2.tag));
  }
  function lf(e, n2) {
    switch (Mi(n2), n2.tag) {
      case 1:
        return le2(n2.type) && Cr(), e = n2.flags, e & 65536 ? (n2.flags = e & -65537 | 128, n2) : null;
      case 3:
        return Kn(), M2(re), M2(J), Ai(), e = n2.flags, (e & 65536) !== 0 && (e & 128) === 0 ? (n2.flags = e & -65537 | 128, n2) : null;
      case 5:
        return Vi(n2), null;
      case 13:
        if (M2(O3), e = n2.memoizedState, e !== null && e.dehydrated !== null) {
          if (n2.alternate === null) throw Error(v2(340));
          Qn();
        }
        return e = n2.flags, e & 65536 ? (n2.flags = e & -65537 | 128, n2) : null;
      case 19:
        return M2(O3), null;
      case 4:
        return Kn(), null;
      case 10:
        return Fi(n2.type._context), null;
      case 22:
      case 23:
        return Ji(), null;
      case 24:
        return null;
      default:
        return null;
    }
  }
  var tr = false, Z2 = false, uf = typeof WeakSet == "function" ? WeakSet : Set, w2 = null;
  function Rn(e, n2) {
    var t = e.ref;
    if (t !== null) if (typeof t == "function") try {
      t(null);
    } catch (r) {
      I2(e, n2, r);
    }
    else t.current = null;
  }
  function ui(e, n2, t) {
    try {
      t();
    } catch (r) {
      I2(e, n2, r);
    }
  }
  var ro = false;
  function of(e, n2) {
    if (Wl = wr, e = bo(), Li(e)) {
      if ("selectionStart" in e) var t = { start: e.selectionStart, end: e.selectionEnd };
      else e: {
        t = (t = e.ownerDocument) && t.defaultView || globalThis;
        var r = t.getSelection && t.getSelection();
        if (r && r.rangeCount !== 0) {
          t = r.anchorNode;
          var l3 = r.anchorOffset, i2 = r.focusNode;
          r = r.focusOffset;
          try {
            t.nodeType, i2.nodeType;
          } catch {
            t = null;
            break e;
          }
          var u2 = 0, o = -1, s = -1, d3 = 0, m2 = 0, h3 = e, p = null;
          n: for (; ; ) {
            for (var g2; h3 !== t || l3 !== 0 && h3.nodeType !== 3 || (o = u2 + l3), h3 !== i2 || r !== 0 && h3.nodeType !== 3 || (s = u2 + r), h3.nodeType === 3 && (u2 += h3.nodeValue.length), (g2 = h3.firstChild) !== null; ) p = h3, h3 = g2;
            for (; ; ) {
              if (h3 === e) break n;
              if (p === t && ++d3 === l3 && (o = u2), p === i2 && ++m2 === r && (s = u2), (g2 = h3.nextSibling) !== null) break;
              h3 = p, p = h3.parentNode;
            }
            h3 = g2;
          }
          t = o === -1 || s === -1 ? null : { start: o, end: s };
        } else t = null;
      }
      t = t || { start: 0, end: 0 };
    } else t = null;
    for (Ql = { focusedElem: e, selectionRange: t }, wr = false, w2 = n2; w2 !== null; ) if (n2 = w2, e = n2.child, (n2.subtreeFlags & 1028) !== 0 && e !== null) e.return = n2, w2 = e;
    else for (; w2 !== null; ) {
      n2 = w2;
      try {
        var S2 = n2.alternate;
        if ((n2.flags & 1024) !== 0) switch (n2.tag) {
          case 0:
          case 11:
          case 15:
            break;
          case 1:
            if (S2 !== null) {
              var k3 = S2.memoizedProps, j2 = S2.memoizedState, c3 = n2.stateNode, a2 = c3.getSnapshotBeforeUpdate(n2.elementType === n2.type ? k3 : we3(n2.type, k3), j2);
              c3.__reactInternalSnapshotBeforeUpdate = a2;
            }
            break;
          case 3:
            var f3 = n2.stateNode.containerInfo;
            f3.nodeType === 1 ? f3.textContent = "" : f3.nodeType === 9 && f3.documentElement && f3.removeChild(f3.documentElement);
            break;
          case 5:
          case 6:
          case 4:
          case 17:
            break;
          default:
            throw Error(v2(163));
        }
      } catch (y3) {
        I2(n2, n2.return, y3);
      }
      if (e = n2.sibling, e !== null) {
        e.return = n2.return, w2 = e;
        break;
      }
      w2 = n2.return;
    }
    return S2 = ro, ro = false, S2;
  }
  function ht(e, n2, t) {
    var r = n2.updateQueue;
    if (r = r !== null ? r.lastEffect : null, r !== null) {
      var l3 = r = r.next;
      do {
        if ((l3.tag & e) === e) {
          var i2 = l3.destroy;
          l3.destroy = void 0, i2 !== void 0 && ui(n2, t, i2);
        }
        l3 = l3.next;
      } while (l3 !== r);
    }
  }
  function Kr(e, n2) {
    if (n2 = n2.updateQueue, n2 = n2 !== null ? n2.lastEffect : null, n2 !== null) {
      var t = n2 = n2.next;
      do {
        if ((t.tag & e) === e) {
          var r = t.create;
          t.destroy = r();
        }
        t = t.next;
      } while (t !== n2);
    }
  }
  function oi(e) {
    var n2 = e.ref;
    if (n2 !== null) {
      var t = e.stateNode;
      switch (e.tag) {
        case 5:
          e = t;
          break;
        default:
          e = t;
      }
      typeof n2 == "function" ? n2(e) : n2.current = e;
    }
  }
  function Gs(e) {
    var n2 = e.alternate;
    n2 !== null && (e.alternate = null, Gs(n2)), e.child = null, e.deletions = null, e.sibling = null, e.tag === 5 && (n2 = e.stateNode, n2 !== null && (delete n2[ze2], delete n2[Pt], delete n2[Yl], delete n2[Hc], delete n2[Wc])), e.stateNode = null, e.return = null, e.dependencies = null, e.memoizedProps = null, e.memoizedState = null, e.pendingProps = null, e.stateNode = null, e.updateQueue = null;
  }
  function Zs(e) {
    return e.tag === 5 || e.tag === 3 || e.tag === 4;
  }
  function lo(e) {
    e: for (; ; ) {
      for (; e.sibling === null; ) {
        if (e.return === null || Zs(e.return)) return null;
        e = e.return;
      }
      for (e.sibling.return = e.return, e = e.sibling; e.tag !== 5 && e.tag !== 6 && e.tag !== 18; ) {
        if (e.flags & 2 || e.child === null || e.tag === 4) continue e;
        e.child.return = e, e = e.child;
      }
      if (!(e.flags & 2)) return e.stateNode;
    }
  }
  function si(e, n2, t) {
    var r = e.tag;
    if (r === 5 || r === 6) e = e.stateNode, n2 ? t.nodeType === 8 ? t.parentNode.insertBefore(e, n2) : t.insertBefore(e, n2) : (t.nodeType === 8 ? (n2 = t.parentNode, n2.insertBefore(e, t)) : (n2 = t, n2.appendChild(e)), t = t._reactRootContainer, t != null || n2.onclick !== null || (n2.onclick = Er));
    else if (r !== 4 && (e = e.child, e !== null)) for (si(e, n2, t), e = e.sibling; e !== null; ) si(e, n2, t), e = e.sibling;
  }
  function ai(e, n2, t) {
    var r = e.tag;
    if (r === 5 || r === 6) e = e.stateNode, n2 ? t.insertBefore(e, n2) : t.appendChild(e);
    else if (r !== 4 && (e = e.child, e !== null)) for (ai(e, n2, t), e = e.sibling; e !== null; ) ai(e, n2, t), e = e.sibling;
  }
  var $2 = null, Se2 = false;
  function Be2(e, n2, t) {
    for (t = t.child; t !== null; ) Js(e, n2, t), t = t.sibling;
  }
  function Js(e, n2, t) {
    if (Pe3 && typeof Pe3.onCommitFiberUnmount == "function") try {
      Pe3.onCommitFiberUnmount(Ur, t);
    } catch {
    }
    switch (t.tag) {
      case 5:
        Z2 || Rn(t, n2);
      case 6:
        var r = $2, l3 = Se2;
        $2 = null, Be2(e, n2, t), $2 = r, Se2 = l3, $2 !== null && (Se2 ? (e = $2, t = t.stateNode, e.nodeType === 8 ? e.parentNode.removeChild(t) : e.removeChild(t)) : $2.removeChild(t.stateNode));
        break;
      case 18:
        $2 !== null && (Se2 ? (e = $2, t = t.stateNode, e.nodeType === 8 ? dl(e.parentNode, t) : e.nodeType === 1 && dl(e, t), Ct(e)) : dl($2, t.stateNode));
        break;
      case 4:
        r = $2, l3 = Se2, $2 = t.stateNode.containerInfo, Se2 = true, Be2(e, n2, t), $2 = r, Se2 = l3;
        break;
      case 0:
      case 11:
      case 14:
      case 15:
        if (!Z2 && (r = t.updateQueue, r !== null && (r = r.lastEffect, r !== null))) {
          l3 = r = r.next;
          do {
            var i2 = l3, u2 = i2.destroy;
            i2 = i2.tag, u2 !== void 0 && ((i2 & 2) !== 0 || (i2 & 4) !== 0) && ui(t, n2, u2), l3 = l3.next;
          } while (l3 !== r);
        }
        Be2(e, n2, t);
        break;
      case 1:
        if (!Z2 && (Rn(t, n2), r = t.stateNode, typeof r.componentWillUnmount == "function")) try {
          r.props = t.memoizedProps, r.state = t.memoizedState, r.componentWillUnmount();
        } catch (o) {
          I2(t, n2, o);
        }
        Be2(e, n2, t);
        break;
      case 21:
        Be2(e, n2, t);
        break;
      case 22:
        t.mode & 1 ? (Z2 = (r = Z2) || t.memoizedState !== null, Be2(e, n2, t), Z2 = r) : Be2(e, n2, t);
        break;
      default:
        Be2(e, n2, t);
    }
  }
  function io(e) {
    var n2 = e.updateQueue;
    if (n2 !== null) {
      e.updateQueue = null;
      var t = e.stateNode;
      t === null && (t = e.stateNode = new uf()), n2.forEach(function(r) {
        var l3 = vf.bind(null, e, r);
        t.has(r) || (t.add(r), r.then(l3, l3));
      });
    }
  }
  function ge3(e, n2) {
    var t = n2.deletions;
    if (t !== null) for (var r = 0; r < t.length; r++) {
      var l3 = t[r];
      try {
        var i2 = e, u2 = n2, o = u2;
        e: for (; o !== null; ) {
          switch (o.tag) {
            case 5:
              $2 = o.stateNode, Se2 = false;
              break e;
            case 3:
              $2 = o.stateNode.containerInfo, Se2 = true;
              break e;
            case 4:
              $2 = o.stateNode.containerInfo, Se2 = true;
              break e;
          }
          o = o.return;
        }
        if ($2 === null) throw Error(v2(160));
        Js(i2, u2, l3), $2 = null, Se2 = false;
        var s = l3.alternate;
        s !== null && (s.return = null), l3.return = null;
      } catch (d3) {
        I2(l3, n2, d3);
      }
    }
    if (n2.subtreeFlags & 12854) for (n2 = n2.child; n2 !== null; ) qs(n2, e), n2 = n2.sibling;
  }
  function qs(e, n2) {
    var t = e.alternate, r = e.flags;
    switch (e.tag) {
      case 0:
      case 11:
      case 14:
      case 15:
        if (ge3(n2, e), Ne2(e), r & 4) {
          try {
            ht(3, e, e.return), Kr(3, e);
          } catch (k3) {
            I2(e, e.return, k3);
          }
          try {
            ht(5, e, e.return);
          } catch (k3) {
            I2(e, e.return, k3);
          }
        }
        break;
      case 1:
        ge3(n2, e), Ne2(e), r & 512 && t !== null && Rn(t, t.return);
        break;
      case 5:
        if (ge3(n2, e), Ne2(e), r & 512 && t !== null && Rn(t, t.return), e.flags & 32) {
          var l3 = e.stateNode;
          try {
            wt(l3, "");
          } catch (k3) {
            I2(e, e.return, k3);
          }
        }
        if (r & 4 && (l3 = e.stateNode, l3 != null)) {
          var i2 = e.memoizedProps, u2 = t !== null ? t.memoizedProps : i2, o = e.type, s = e.updateQueue;
          if (e.updateQueue = null, s !== null) try {
            o === "input" && i2.type === "radio" && i2.name != null && So(l3, i2), Ol(o, u2);
            var d3 = Ol(o, i2);
            for (u2 = 0; u2 < s.length; u2 += 2) {
              var m2 = s[u2], h3 = s[u2 + 1];
              m2 === "style" ? No(l3, h3) : m2 === "dangerouslySetInnerHTML" ? Co(l3, h3) : m2 === "children" ? wt(l3, h3) : vi(l3, m2, h3, d3);
            }
            switch (o) {
              case "input":
                Pl(l3, i2);
                break;
              case "textarea":
                ko(l3, i2);
                break;
              case "select":
                var p = l3._wrapperState.wasMultiple;
                l3._wrapperState.wasMultiple = !!i2.multiple;
                var g2 = i2.value;
                g2 != null ? In(l3, !!i2.multiple, g2, false) : p !== !!i2.multiple && (i2.defaultValue != null ? In(l3, !!i2.multiple, i2.defaultValue, true) : In(l3, !!i2.multiple, i2.multiple ? [] : "", false));
            }
            l3[Pt] = i2;
          } catch (k3) {
            I2(e, e.return, k3);
          }
        }
        break;
      case 6:
        if (ge3(n2, e), Ne2(e), r & 4) {
          if (e.stateNode === null) throw Error(v2(162));
          l3 = e.stateNode, i2 = e.memoizedProps;
          try {
            l3.nodeValue = i2;
          } catch (k3) {
            I2(e, e.return, k3);
          }
        }
        break;
      case 3:
        if (ge3(n2, e), Ne2(e), r & 4 && t !== null && t.memoizedState.isDehydrated) try {
          Ct(n2.containerInfo);
        } catch (k3) {
          I2(e, e.return, k3);
        }
        break;
      case 4:
        ge3(n2, e), Ne2(e);
        break;
      case 13:
        ge3(n2, e), Ne2(e), l3 = e.child, l3.flags & 8192 && (i2 = l3.memoizedState !== null, l3.stateNode.isHidden = i2, !i2 || l3.alternate !== null && l3.alternate.memoizedState !== null || (Gi = U3())), r & 4 && io(e);
        break;
      case 22:
        if (m2 = t !== null && t.memoizedState !== null, e.mode & 1 ? (Z2 = (d3 = Z2) || m2, ge3(n2, e), Z2 = d3) : ge3(n2, e), Ne2(e), r & 8192) {
          if (d3 = e.memoizedState !== null, (e.stateNode.isHidden = d3) && !m2 && (e.mode & 1) !== 0) for (w2 = e, m2 = e.child; m2 !== null; ) {
            for (h3 = w2 = m2; w2 !== null; ) {
              switch (p = w2, g2 = p.child, p.tag) {
                case 0:
                case 11:
                case 14:
                case 15:
                  ht(4, p, p.return);
                  break;
                case 1:
                  Rn(p, p.return);
                  var S2 = p.stateNode;
                  if (typeof S2.componentWillUnmount == "function") {
                    r = p, t = p.return;
                    try {
                      n2 = r, S2.props = n2.memoizedProps, S2.state = n2.memoizedState, S2.componentWillUnmount();
                    } catch (k3) {
                      I2(r, t, k3);
                    }
                  }
                  break;
                case 5:
                  Rn(p, p.return);
                  break;
                case 22:
                  if (p.memoizedState !== null) {
                    oo(h3);
                    continue;
                  }
              }
              g2 !== null ? (g2.return = p, w2 = g2) : oo(h3);
            }
            m2 = m2.sibling;
          }
          e: for (m2 = null, h3 = e; ; ) {
            if (h3.tag === 5) {
              if (m2 === null) {
                m2 = h3;
                try {
                  l3 = h3.stateNode, d3 ? (i2 = l3.style, typeof i2.setProperty == "function" ? i2.setProperty("display", "none", "important") : i2.display = "none") : (o = h3.stateNode, s = h3.memoizedProps.style, u2 = s != null && s.hasOwnProperty("display") ? s.display : null, o.style.display = xo("display", u2));
                } catch (k3) {
                  I2(e, e.return, k3);
                }
              }
            } else if (h3.tag === 6) {
              if (m2 === null) try {
                h3.stateNode.nodeValue = d3 ? "" : h3.memoizedProps;
              } catch (k3) {
                I2(e, e.return, k3);
              }
            } else if ((h3.tag !== 22 && h3.tag !== 23 || h3.memoizedState === null || h3 === e) && h3.child !== null) {
              h3.child.return = h3, h3 = h3.child;
              continue;
            }
            if (h3 === e) break e;
            for (; h3.sibling === null; ) {
              if (h3.return === null || h3.return === e) break e;
              m2 === h3 && (m2 = null), h3 = h3.return;
            }
            m2 === h3 && (m2 = null), h3.sibling.return = h3.return, h3 = h3.sibling;
          }
        }
        break;
      case 19:
        ge3(n2, e), Ne2(e), r & 4 && io(e);
        break;
      case 21:
        break;
      default:
        ge3(n2, e), Ne2(e);
    }
  }
  function Ne2(e) {
    var n2 = e.flags;
    if (n2 & 2) {
      try {
        e: {
          for (var t = e.return; t !== null; ) {
            if (Zs(t)) {
              var r = t;
              break e;
            }
            t = t.return;
          }
          throw Error(v2(160));
        }
        switch (r.tag) {
          case 5:
            var l3 = r.stateNode;
            r.flags & 32 && (wt(l3, ""), r.flags &= -33);
            var i2 = lo(e);
            ai(e, i2, l3);
            break;
          case 3:
          case 4:
            var u2 = r.stateNode.containerInfo, o = lo(e);
            si(e, o, u2);
            break;
          default:
            throw Error(v2(161));
        }
      } catch (s) {
        I2(e, e.return, s);
      }
      e.flags &= -3;
    }
    n2 & 4096 && (e.flags &= -4097);
  }
  function sf(e, n2, t) {
    w2 = e, bs(e, n2, t);
  }
  function bs(e, n2, t) {
    for (var r = (e.mode & 1) !== 0; w2 !== null; ) {
      var l3 = w2, i2 = l3.child;
      if (l3.tag === 22 && r) {
        var u2 = l3.memoizedState !== null || tr;
        if (!u2) {
          var o = l3.alternate, s = o !== null && o.memoizedState !== null || Z2;
          o = tr;
          var d3 = Z2;
          if (tr = u2, (Z2 = s) && !d3) for (w2 = l3; w2 !== null; ) u2 = w2, s = u2.child, u2.tag === 22 && u2.memoizedState !== null ? so(l3) : s !== null ? (s.return = u2, w2 = s) : so(l3);
          for (; i2 !== null; ) w2 = i2, bs(i2, n2, t), i2 = i2.sibling;
          w2 = l3, tr = o, Z2 = d3;
        }
        uo(e, n2, t);
      } else (l3.subtreeFlags & 8772) !== 0 && i2 !== null ? (i2.return = l3, w2 = i2) : uo(e, n2, t);
    }
  }
  function uo(e) {
    for (; w2 !== null; ) {
      var n2 = w2;
      if ((n2.flags & 8772) !== 0) {
        var t = n2.alternate;
        try {
          if ((n2.flags & 8772) !== 0) switch (n2.tag) {
            case 0:
            case 11:
            case 15:
              Z2 || Kr(5, n2);
              break;
            case 1:
              var r = n2.stateNode;
              if (n2.flags & 4 && !Z2) if (t === null) r.componentDidMount();
              else {
                var l3 = n2.elementType === n2.type ? t.memoizedProps : we3(n2.type, t.memoizedProps);
                r.componentDidUpdate(l3, t.memoizedState, r.__reactInternalSnapshotBeforeUpdate);
              }
              var i2 = n2.updateQueue;
              i2 !== null && Qu(n2, i2, r);
              break;
            case 3:
              var u2 = n2.updateQueue;
              if (u2 !== null) {
                if (t = null, n2.child !== null) switch (n2.child.tag) {
                  case 5:
                    t = n2.child.stateNode;
                    break;
                  case 1:
                    t = n2.child.stateNode;
                }
                Qu(n2, u2, t);
              }
              break;
            case 5:
              var o = n2.stateNode;
              if (t === null && n2.flags & 4) {
                t = o;
                var s = n2.memoizedProps;
                switch (n2.type) {
                  case "button":
                  case "input":
                  case "select":
                  case "textarea":
                    s.autoFocus && t.focus();
                    break;
                  case "img":
                    s.src && (t.src = s.src);
                }
              }
              break;
            case 6:
              break;
            case 4:
              break;
            case 12:
              break;
            case 13:
              if (n2.memoizedState === null) {
                var d3 = n2.alternate;
                if (d3 !== null) {
                  var m2 = d3.memoizedState;
                  if (m2 !== null) {
                    var h3 = m2.dehydrated;
                    h3 !== null && Ct(h3);
                  }
                }
              }
              break;
            case 19:
            case 17:
            case 21:
            case 22:
            case 23:
            case 25:
              break;
            default:
              throw Error(v2(163));
          }
          Z2 || n2.flags & 512 && oi(n2);
        } catch (p) {
          I2(n2, n2.return, p);
        }
      }
      if (n2 === e) {
        w2 = null;
        break;
      }
      if (t = n2.sibling, t !== null) {
        t.return = n2.return, w2 = t;
        break;
      }
      w2 = n2.return;
    }
  }
  function oo(e) {
    for (; w2 !== null; ) {
      var n2 = w2;
      if (n2 === e) {
        w2 = null;
        break;
      }
      var t = n2.sibling;
      if (t !== null) {
        t.return = n2.return, w2 = t;
        break;
      }
      w2 = n2.return;
    }
  }
  function so(e) {
    for (; w2 !== null; ) {
      var n2 = w2;
      try {
        switch (n2.tag) {
          case 0:
          case 11:
          case 15:
            var t = n2.return;
            try {
              Kr(4, n2);
            } catch (s) {
              I2(n2, t, s);
            }
            break;
          case 1:
            var r = n2.stateNode;
            if (typeof r.componentDidMount == "function") {
              var l3 = n2.return;
              try {
                r.componentDidMount();
              } catch (s) {
                I2(n2, l3, s);
              }
            }
            var i2 = n2.return;
            try {
              oi(n2);
            } catch (s) {
              I2(n2, i2, s);
            }
            break;
          case 5:
            var u2 = n2.return;
            try {
              oi(n2);
            } catch (s) {
              I2(n2, u2, s);
            }
        }
      } catch (s) {
        I2(n2, n2.return, s);
      }
      if (n2 === e) {
        w2 = null;
        break;
      }
      var o = n2.sibling;
      if (o !== null) {
        o.return = n2.return, w2 = o;
        break;
      }
      w2 = n2.return;
    }
  }
  var af = Math.ceil, Or = Ve2.ReactCurrentDispatcher, Yi = Ve2.ReactCurrentOwner, he3 = Ve2.ReactCurrentBatchConfig, _2 = 0, Q = null, V2 = null, K2 = 0, ue2 = 0, Fn = un(0), B3 = 0, Rt = null, gn = 0, Yr = 0, Xi = 0, vt = null, ne2 = null, Gi = 0, Xn = 1 / 0, Te2 = null, Rr = false, ci = null, be3 = null, rr = false, Ye = null, Fr = 0, yt = 0, fi = null, fr = -1, dr = 0;
  function b() {
    return (_2 & 6) !== 0 ? U3() : fr !== -1 ? fr : fr = U3();
  }
  function en(e) {
    return (e.mode & 1) === 0 ? 1 : (_2 & 2) !== 0 && K2 !== 0 ? K2 & -K2 : $c.transition !== null ? (dr === 0 && (dr = jo()), dr) : (e = P, e !== 0 || (e = globalThis.event, e = e === void 0 ? 16 : Qo(e.type)), e);
  }
  function Ce2(e, n2, t, r) {
    if (50 < yt) throw yt = 0, fi = null, Error(v2(185));
    Ft(e, t, r), ((_2 & 2) === 0 || e !== Q) && (e === Q && ((_2 & 2) === 0 && (Yr |= t), B3 === 4 && $e2(e, K2)), ie(e, r), t === 1 && _2 === 0 && (n2.mode & 1) === 0 && (Xn = U3() + 500, Wr && on()));
  }
  function ie(e, n2) {
    var t = e.callbackNode;
    Ya(e, n2);
    var r = gr(e, e === Q ? K2 : 0);
    if (r === 0) t !== null && vu(t), e.callbackNode = null, e.callbackPriority = 0;
    else if (n2 = r & -r, e.callbackPriority !== n2) {
      if (t != null && vu(t), n2 === 1) e.tag === 0 ? Qc(ao.bind(null, e)) : as(ao.bind(null, e)), Ac(function() {
        (_2 & 6) === 0 && on();
      }), t = null;
      else {
        switch (Uo(r)) {
          case 1:
            t = ki;
            break;
          case 4:
            t = Fo;
            break;
          case 16:
            t = yr;
            break;
          case 536870912:
            t = Io;
            break;
          default:
            t = yr;
        }
        t = oa(t, ea.bind(null, e));
      }
      e.callbackPriority = n2, e.callbackNode = t;
    }
  }
  function ea(e, n2) {
    if (fr = -1, dr = 0, (_2 & 6) !== 0) throw Error(v2(327));
    var t = e.callbackNode;
    if (Bn() && e.callbackNode !== t) return null;
    var r = gr(e, e === Q ? K2 : 0);
    if (r === 0) return null;
    if ((r & 30) !== 0 || (r & e.expiredLanes) !== 0 || n2) n2 = Ir(e, r);
    else {
      n2 = r;
      var l3 = _2;
      _2 |= 2;
      var i2 = ta();
      (Q !== e || K2 !== n2) && (Te2 = null, Xn = U3() + 500, pn(e, n2));
      do
        try {
          df();
          break;
        } catch (o) {
          na(e, o);
        }
      while (true);
      Ri(), Or.current = i2, _2 = l3, V2 !== null ? n2 = 0 : (Q = null, K2 = 0, n2 = B3);
    }
    if (n2 !== 0) {
      if (n2 === 2 && (l3 = Ul(e), l3 !== 0 && (r = l3, n2 = di(e, l3))), n2 === 1) throw t = Rt, pn(e, 0), $e2(e, r), ie(e, U3()), t;
      if (n2 === 6) $e2(e, r);
      else {
        if (l3 = e.current.alternate, (r & 30) === 0 && !cf(l3) && (n2 = Ir(e, r), n2 === 2 && (i2 = Ul(e), i2 !== 0 && (r = i2, n2 = di(e, i2))), n2 === 1)) throw t = Rt, pn(e, 0), $e2(e, r), ie(e, U3()), t;
        switch (e.finishedWork = l3, e.finishedLanes = r, n2) {
          case 0:
          case 1:
            throw Error(v2(345));
          case 2:
            an(e, ne2, Te2);
            break;
          case 3:
            if ($e2(e, r), (r & 130023424) === r && (n2 = Gi + 500 - U3(), 10 < n2)) {
              if (gr(e, 0) !== 0) break;
              if (l3 = e.suspendedLanes, (l3 & r) !== r) {
                b(), e.pingedLanes |= e.suspendedLanes & l3;
                break;
              }
              e.timeoutHandle = Kl(an.bind(null, e, ne2, Te2), n2);
              break;
            }
            an(e, ne2, Te2);
            break;
          case 4:
            if ($e2(e, r), (r & 4194240) === r) break;
            for (n2 = e.eventTimes, l3 = -1; 0 < r; ) {
              var u2 = 31 - Ee2(r);
              i2 = 1 << u2, u2 = n2[u2], u2 > l3 && (l3 = u2), r &= ~i2;
            }
            if (r = l3, r = U3() - r, r = (120 > r ? 120 : 480 > r ? 480 : 1080 > r ? 1080 : 1920 > r ? 1920 : 3e3 > r ? 3e3 : 4320 > r ? 4320 : 1960 * af(r / 1960)) - r, 10 < r) {
              e.timeoutHandle = Kl(an.bind(null, e, ne2, Te2), r);
              break;
            }
            an(e, ne2, Te2);
            break;
          case 5:
            an(e, ne2, Te2);
            break;
          default:
            throw Error(v2(329));
        }
      }
    }
    return ie(e, U3()), e.callbackNode === t ? ea.bind(null, e) : null;
  }
  function di(e, n2) {
    var t = vt;
    return e.current.memoizedState.isDehydrated && (pn(e, n2).flags |= 256), e = Ir(e, n2), e !== 2 && (n2 = ne2, ne2 = t, n2 !== null && pi(n2)), e;
  }
  function pi(e) {
    ne2 === null ? ne2 = e : ne2.push.apply(ne2, e);
  }
  function cf(e) {
    for (var n2 = e; ; ) {
      if (n2.flags & 16384) {
        var t = n2.updateQueue;
        if (t !== null && (t = t.stores, t !== null)) for (var r = 0; r < t.length; r++) {
          var l3 = t[r], i2 = l3.getSnapshot;
          l3 = l3.value;
          try {
            if (!xe3(i2(), l3)) return false;
          } catch {
            return false;
          }
        }
      }
      if (t = n2.child, n2.subtreeFlags & 16384 && t !== null) t.return = n2, n2 = t;
      else {
        if (n2 === e) break;
        for (; n2.sibling === null; ) {
          if (n2.return === null || n2.return === e) return true;
          n2 = n2.return;
        }
        n2.sibling.return = n2.return, n2 = n2.sibling;
      }
    }
    return true;
  }
  function $e2(e, n2) {
    for (n2 &= ~Xi, n2 &= ~Yr, e.suspendedLanes |= n2, e.pingedLanes &= ~n2, e = e.expirationTimes; 0 < n2; ) {
      var t = 31 - Ee2(n2), r = 1 << t;
      e[t] = -1, n2 &= ~r;
    }
  }
  function ao(e) {
    if ((_2 & 6) !== 0) throw Error(v2(327));
    Bn();
    var n2 = gr(e, 0);
    if ((n2 & 1) === 0) return ie(e, U3()), null;
    var t = Ir(e, n2);
    if (e.tag !== 0 && t === 2) {
      var r = Ul(e);
      r !== 0 && (n2 = r, t = di(e, r));
    }
    if (t === 1) throw t = Rt, pn(e, 0), $e2(e, n2), ie(e, U3()), t;
    if (t === 6) throw Error(v2(345));
    return e.finishedWork = e.current.alternate, e.finishedLanes = n2, an(e, ne2, Te2), ie(e, U3()), null;
  }
  function Zi(e, n2) {
    var t = _2;
    _2 |= 1;
    try {
      return e(n2);
    } finally {
      _2 = t, _2 === 0 && (Xn = U3() + 500, Wr && on());
    }
  }
  function wn(e) {
    Ye !== null && Ye.tag === 0 && (_2 & 6) === 0 && Bn();
    var n2 = _2;
    _2 |= 1;
    var t = he3.transition, r = P;
    try {
      if (he3.transition = null, P = 1, e) return e();
    } finally {
      P = r, he3.transition = t, _2 = n2, (_2 & 6) === 0 && on();
    }
  }
  function Ji() {
    ue2 = Fn.current, M2(Fn);
  }
  function pn(e, n2) {
    e.finishedWork = null, e.finishedLanes = 0;
    var t = e.timeoutHandle;
    if (t !== -1 && (e.timeoutHandle = -1, Vc(t)), V2 !== null) for (t = V2.return; t !== null; ) {
      var r = t;
      switch (Mi(r), r.tag) {
        case 1:
          r = r.type.childContextTypes, r != null && Cr();
          break;
        case 3:
          Kn(), M2(re), M2(J), Ai();
          break;
        case 5:
          Vi(r);
          break;
        case 4:
          Kn();
          break;
        case 13:
          M2(O3);
          break;
        case 19:
          M2(O3);
          break;
        case 10:
          Fi(r.type._context);
          break;
        case 22:
        case 23:
          Ji();
      }
      t = t.return;
    }
    if (Q = e, V2 = e = nn(e.current, null), K2 = ue2 = n2, B3 = 0, Rt = null, Xi = Yr = gn = 0, ne2 = vt = null, fn !== null) {
      for (n2 = 0; n2 < fn.length; n2++) if (t = fn[n2], r = t.interleaved, r !== null) {
        t.interleaved = null;
        var l3 = r.next, i2 = t.pending;
        if (i2 !== null) {
          var u2 = i2.next;
          i2.next = l3, r.next = u2;
        }
        t.pending = r;
      }
      fn = null;
    }
    return e;
  }
  function na(e, n2) {
    do {
      var t = V2;
      try {
        if (Ri(), sr.current = Dr, Mr) {
          for (var r = R2.memoizedState; r !== null; ) {
            var l3 = r.queue;
            l3 !== null && (l3.pending = null), r = r.next;
          }
          Mr = false;
        }
        if (yn = 0, W = A2 = R2 = null, mt = false, Mt = 0, Yi.current = null, t === null || t.return === null) {
          B3 = 1, Rt = n2, V2 = null;
          break;
        }
        e: {
          var i2 = e, u2 = t.return, o = t, s = n2;
          if (n2 = K2, o.flags |= 32768, s !== null && typeof s == "object" && typeof s.then == "function") {
            var d3 = s, m2 = o, h3 = m2.tag;
            if ((m2.mode & 1) === 0 && (h3 === 0 || h3 === 11 || h3 === 15)) {
              var p = m2.alternate;
              p ? (m2.updateQueue = p.updateQueue, m2.memoizedState = p.memoizedState, m2.lanes = p.lanes) : (m2.updateQueue = null, m2.memoizedState = null);
            }
            var g2 = Zu(u2);
            if (g2 !== null) {
              g2.flags &= -257, Ju(g2, u2, o, i2, n2), g2.mode & 1 && Gu(i2, d3, n2), n2 = g2, s = d3;
              var S2 = n2.updateQueue;
              if (S2 === null) {
                var k3 = /* @__PURE__ */ new Set();
                k3.add(s), n2.updateQueue = k3;
              } else S2.add(s);
              break e;
            } else {
              if ((n2 & 1) === 0) {
                Gu(i2, d3, n2), qi();
                break e;
              }
              s = Error(v2(426));
            }
          } else if (D2 && o.mode & 1) {
            var j2 = Zu(u2);
            if (j2 !== null) {
              (j2.flags & 65536) === 0 && (j2.flags |= 256), Ju(j2, u2, o, i2, n2), Di(Yn(s, o));
              break e;
            }
          }
          i2 = s = Yn(s, o), B3 !== 4 && (B3 = 2), vt === null ? vt = [i2] : vt.push(i2), i2 = u2;
          do {
            switch (i2.tag) {
              case 3:
                i2.flags |= 65536, n2 &= -n2, i2.lanes |= n2;
                var c3 = Us(i2, s, n2);
                Wu(i2, c3);
                break e;
              case 1:
                o = s;
                var a2 = i2.type, f3 = i2.stateNode;
                if ((i2.flags & 128) === 0 && (typeof a2.getDerivedStateFromError == "function" || f3 !== null && typeof f3.componentDidCatch == "function" && (be3 === null || !be3.has(f3)))) {
                  i2.flags |= 65536, n2 &= -n2, i2.lanes |= n2;
                  var y3 = Vs(i2, o, n2);
                  Wu(i2, y3);
                  break e;
                }
            }
            i2 = i2.return;
          } while (i2 !== null);
        }
        la(t);
      } catch (E3) {
        n2 = E3, V2 === t && t !== null && (V2 = t = t.return);
        continue;
      }
      break;
    } while (true);
  }
  function ta() {
    var e = Or.current;
    return Or.current = Dr, e === null ? Dr : e;
  }
  function qi() {
    (B3 === 0 || B3 === 3 || B3 === 2) && (B3 = 4), Q === null || (gn & 268435455) === 0 && (Yr & 268435455) === 0 || $e2(Q, K2);
  }
  function Ir(e, n2) {
    var t = _2;
    _2 |= 2;
    var r = ta();
    (Q !== e || K2 !== n2) && (Te2 = null, pn(e, n2));
    do
      try {
        ff();
        break;
      } catch (l3) {
        na(e, l3);
      }
    while (true);
    if (Ri(), _2 = t, Or.current = r, V2 !== null) throw Error(v2(261));
    return Q = null, K2 = 0, B3;
  }
  function ff() {
    for (; V2 !== null; ) ra(V2);
  }
  function df() {
    for (; V2 !== null && !Ua(); ) ra(V2);
  }
  function ra(e) {
    var n2 = ua(e.alternate, e, ue2);
    e.memoizedProps = e.pendingProps, n2 === null ? la(e) : V2 = n2, Yi.current = null;
  }
  function la(e) {
    var n2 = e;
    do {
      var t = n2.alternate;
      if (e = n2.return, (n2.flags & 32768) === 0) {
        if (t = rf(t, n2, ue2), t !== null) {
          V2 = t;
          return;
        }
      } else {
        if (t = lf(t, n2), t !== null) {
          t.flags &= 32767, V2 = t;
          return;
        }
        if (e !== null) e.flags |= 32768, e.subtreeFlags = 0, e.deletions = null;
        else {
          B3 = 6, V2 = null;
          return;
        }
      }
      if (n2 = n2.sibling, n2 !== null) {
        V2 = n2;
        return;
      }
      V2 = n2 = e;
    } while (n2 !== null);
    B3 === 0 && (B3 = 5);
  }
  function an(e, n2, t) {
    var r = P, l3 = he3.transition;
    try {
      he3.transition = null, P = 1, pf(e, n2, t, r);
    } finally {
      he3.transition = l3, P = r;
    }
    return null;
  }
  function pf(e, n2, t, r) {
    do
      Bn();
    while (Ye !== null);
    if ((_2 & 6) !== 0) throw Error(v2(327));
    t = e.finishedWork;
    var l3 = e.finishedLanes;
    if (t === null) return null;
    if (e.finishedWork = null, e.finishedLanes = 0, t === e.current) throw Error(v2(177));
    e.callbackNode = null, e.callbackPriority = 0;
    var i2 = t.lanes | t.childLanes;
    if (Xa(e, i2), e === Q && (V2 = Q = null, K2 = 0), (t.subtreeFlags & 2064) === 0 && (t.flags & 2064) === 0 || rr || (rr = true, oa(yr, function() {
      return Bn(), null;
    })), i2 = (t.flags & 15990) !== 0, (t.subtreeFlags & 15990) !== 0 || i2) {
      i2 = he3.transition, he3.transition = null;
      var u2 = P;
      P = 1;
      var o = _2;
      _2 |= 4, Yi.current = null, of(e, t), qs(t, e), Rc(Ql), wr = !!Wl, Ql = Wl = null, e.current = t, sf(t, e, l3), Va(), _2 = o, P = u2, he3.transition = i2;
    } else e.current = t;
    if (rr && (rr = false, Ye = e, Fr = l3), i2 = e.pendingLanes, i2 === 0 && (be3 = null), Ha(t.stateNode, r), ie(e, U3()), n2 !== null) for (r = e.onRecoverableError, t = 0; t < n2.length; t++) l3 = n2[t], r(l3.value, { componentStack: l3.stack, digest: l3.digest });
    if (Rr) throw Rr = false, e = ci, ci = null, e;
    return (Fr & 1) !== 0 && e.tag !== 0 && Bn(), i2 = e.pendingLanes, (i2 & 1) !== 0 ? e === fi ? yt++ : (yt = 0, fi = e) : yt = 0, on(), null;
  }
  function Bn() {
    if (Ye !== null) {
      var e = Uo(Fr), n2 = he3.transition, t = P;
      try {
        if (he3.transition = null, P = 16 > e ? 16 : e, Ye === null) var r = false;
        else {
          if (e = Ye, Ye = null, Fr = 0, (_2 & 6) !== 0) throw Error(v2(331));
          var l3 = _2;
          for (_2 |= 4, w2 = e.current; w2 !== null; ) {
            var i2 = w2, u2 = i2.child;
            if ((w2.flags & 16) !== 0) {
              var o = i2.deletions;
              if (o !== null) {
                for (var s = 0; s < o.length; s++) {
                  var d3 = o[s];
                  for (w2 = d3; w2 !== null; ) {
                    var m2 = w2;
                    switch (m2.tag) {
                      case 0:
                      case 11:
                      case 15:
                        ht(8, m2, i2);
                    }
                    var h3 = m2.child;
                    if (h3 !== null) h3.return = m2, w2 = h3;
                    else for (; w2 !== null; ) {
                      m2 = w2;
                      var p = m2.sibling, g2 = m2.return;
                      if (Gs(m2), m2 === d3) {
                        w2 = null;
                        break;
                      }
                      if (p !== null) {
                        p.return = g2, w2 = p;
                        break;
                      }
                      w2 = g2;
                    }
                  }
                }
                var S2 = i2.alternate;
                if (S2 !== null) {
                  var k3 = S2.child;
                  if (k3 !== null) {
                    S2.child = null;
                    do {
                      var j2 = k3.sibling;
                      k3.sibling = null, k3 = j2;
                    } while (k3 !== null);
                  }
                }
                w2 = i2;
              }
            }
            if ((i2.subtreeFlags & 2064) !== 0 && u2 !== null) u2.return = i2, w2 = u2;
            else e: for (; w2 !== null; ) {
              if (i2 = w2, (i2.flags & 2048) !== 0) switch (i2.tag) {
                case 0:
                case 11:
                case 15:
                  ht(9, i2, i2.return);
              }
              var c3 = i2.sibling;
              if (c3 !== null) {
                c3.return = i2.return, w2 = c3;
                break e;
              }
              w2 = i2.return;
            }
          }
          var a2 = e.current;
          for (w2 = a2; w2 !== null; ) {
            u2 = w2;
            var f3 = u2.child;
            if ((u2.subtreeFlags & 2064) !== 0 && f3 !== null) f3.return = u2, w2 = f3;
            else e: for (u2 = a2; w2 !== null; ) {
              if (o = w2, (o.flags & 2048) !== 0) try {
                switch (o.tag) {
                  case 0:
                  case 11:
                  case 15:
                    Kr(9, o);
                }
              } catch (E3) {
                I2(o, o.return, E3);
              }
              if (o === u2) {
                w2 = null;
                break e;
              }
              var y3 = o.sibling;
              if (y3 !== null) {
                y3.return = o.return, w2 = y3;
                break e;
              }
              w2 = o.return;
            }
          }
          if (_2 = l3, on(), Pe3 && typeof Pe3.onPostCommitFiberRoot == "function") try {
            Pe3.onPostCommitFiberRoot(Ur, e);
          } catch {
          }
          r = true;
        }
        return r;
      } finally {
        P = t, he3.transition = n2;
      }
    }
    return false;
  }
  function co(e, n2, t) {
    n2 = Yn(t, n2), n2 = Us(e, n2, 1), e = qe2(e, n2, 1), n2 = b(), e !== null && (Ft(e, 1, n2), ie(e, n2));
  }
  function I2(e, n2, t) {
    if (e.tag === 3) co(e, e, t);
    else for (; n2 !== null; ) {
      if (n2.tag === 3) {
        co(n2, e, t);
        break;
      } else if (n2.tag === 1) {
        var r = n2.stateNode;
        if (typeof n2.type.getDerivedStateFromError == "function" || typeof r.componentDidCatch == "function" && (be3 === null || !be3.has(r))) {
          e = Yn(t, e), e = Vs(n2, e, 1), n2 = qe2(n2, e, 1), e = b(), n2 !== null && (Ft(n2, 1, e), ie(n2, e));
          break;
        }
      }
      n2 = n2.return;
    }
  }
  function mf(e, n2, t) {
    var r = e.pingCache;
    r !== null && r.delete(n2), n2 = b(), e.pingedLanes |= e.suspendedLanes & t, Q === e && (K2 & t) === t && (B3 === 4 || B3 === 3 && (K2 & 130023424) === K2 && 500 > U3() - Gi ? pn(e, 0) : Xi |= t), ie(e, n2);
  }
  function ia(e, n2) {
    n2 === 0 && ((e.mode & 1) === 0 ? n2 = 1 : (n2 = Wt, Wt <<= 1, (Wt & 130023424) === 0 && (Wt = 4194304)));
    var t = b();
    e = je2(e, n2), e !== null && (Ft(e, n2, t), ie(e, t));
  }
  function hf(e) {
    var n2 = e.memoizedState, t = 0;
    n2 !== null && (t = n2.retryLane), ia(e, t);
  }
  function vf(e, n2) {
    var t = 0;
    switch (e.tag) {
      case 13:
        var r = e.stateNode, l3 = e.memoizedState;
        l3 !== null && (t = l3.retryLane);
        break;
      case 19:
        r = e.stateNode;
        break;
      default:
        throw Error(v2(314));
    }
    r !== null && r.delete(n2), ia(e, t);
  }
  var ua;
  ua = function(e, n2, t) {
    if (e !== null) if (e.memoizedProps !== n2.pendingProps || re.current) te = true;
    else {
      if ((e.lanes & t) === 0 && (n2.flags & 128) === 0) return te = false, tf(e, n2, t);
      te = (e.flags & 131072) !== 0;
    }
    else te = false, D2 && (n2.flags & 1048576) !== 0 && cs(n2, _r, n2.index);
    switch (n2.lanes = 0, n2.tag) {
      case 2:
        var r = n2.type;
        cr(e, n2), e = n2.pendingProps;
        var l3 = Wn(n2, J.current);
        An(n2, t), l3 = Hi(null, n2, r, e, l3, t);
        var i2 = Wi();
        return n2.flags |= 1, typeof l3 == "object" && l3 !== null && typeof l3.render == "function" && l3.$$typeof === void 0 ? (n2.tag = 1, n2.memoizedState = null, n2.updateQueue = null, le2(r) ? (i2 = true, xr(n2)) : i2 = false, n2.memoizedState = l3.state !== null && l3.state !== void 0 ? l3.state : null, ji(n2), l3.updater = $r, n2.stateNode = l3, l3._reactInternals = n2, bl(n2, r, e, t), n2 = ti(null, n2, r, true, i2, t)) : (n2.tag = 0, D2 && i2 && Ti(n2), q2(null, n2, l3, t), n2 = n2.child), n2;
      case 16:
        r = n2.elementType;
        e: {
          switch (cr(e, n2), e = n2.pendingProps, l3 = r._init, r = l3(r._payload), n2.type = r, l3 = n2.tag = gf(r), e = we3(r, e), l3) {
            case 0:
              n2 = ni(null, n2, r, e, t);
              break e;
            case 1:
              n2 = eo(null, n2, r, e, t);
              break e;
            case 11:
              n2 = qu(null, n2, r, e, t);
              break e;
            case 14:
              n2 = bu(null, n2, r, we3(r.type, e), t);
              break e;
          }
          throw Error(v2(306, r, ""));
        }
        return n2;
      case 0:
        return r = n2.type, l3 = n2.pendingProps, l3 = n2.elementType === r ? l3 : we3(r, l3), ni(e, n2, r, l3, t);
      case 1:
        return r = n2.type, l3 = n2.pendingProps, l3 = n2.elementType === r ? l3 : we3(r, l3), eo(e, n2, r, l3, t);
      case 3:
        e: {
          if (Ws(n2), e === null) throw Error(v2(387));
          r = n2.pendingProps, i2 = n2.memoizedState, l3 = i2.element, vs(e, n2), Lr(n2, r, null, t);
          var u2 = n2.memoizedState;
          if (r = u2.element, i2.isDehydrated) if (i2 = { element: r, isDehydrated: false, cache: u2.cache, pendingSuspenseBoundaries: u2.pendingSuspenseBoundaries, transitions: u2.transitions }, n2.updateQueue.baseState = i2, n2.memoizedState = i2, n2.flags & 256) {
            l3 = Yn(Error(v2(423)), n2), n2 = no(e, n2, r, t, l3);
            break e;
          } else if (r !== l3) {
            l3 = Yn(Error(v2(424)), n2), n2 = no(e, n2, r, t, l3);
            break e;
          } else for (oe2 = Je(n2.stateNode.containerInfo.firstChild), se2 = n2, D2 = true, ke3 = null, t = ms(n2, null, r, t), n2.child = t; t; ) t.flags = t.flags & -3 | 4096, t = t.sibling;
          else {
            if (Qn(), r === l3) {
              n2 = Ue2(e, n2, t);
              break e;
            }
            q2(e, n2, r, t);
          }
          n2 = n2.child;
        }
        return n2;
      case 5:
        return ys(n2), e === null && Zl(n2), r = n2.type, l3 = n2.pendingProps, i2 = e !== null ? e.memoizedProps : null, u2 = l3.children, $l(r, l3) ? u2 = null : i2 !== null && $l(r, i2) && (n2.flags |= 32), Hs(e, n2), q2(e, n2, u2, t), n2.child;
      case 6:
        return e === null && Zl(n2), null;
      case 13:
        return Qs(e, n2, t);
      case 4:
        return Ui(n2, n2.stateNode.containerInfo), r = n2.pendingProps, e === null ? n2.child = $n(n2, null, r, t) : q2(e, n2, r, t), n2.child;
      case 11:
        return r = n2.type, l3 = n2.pendingProps, l3 = n2.elementType === r ? l3 : we3(r, l3), qu(e, n2, r, l3, t);
      case 7:
        return q2(e, n2, n2.pendingProps, t), n2.child;
      case 8:
        return q2(e, n2, n2.pendingProps.children, t), n2.child;
      case 12:
        return q2(e, n2, n2.pendingProps.children, t), n2.child;
      case 10:
        e: {
          if (r = n2.type._context, l3 = n2.pendingProps, i2 = n2.memoizedProps, u2 = l3.value, L3(zr, r._currentValue), r._currentValue = u2, i2 !== null) if (xe3(i2.value, u2)) {
            if (i2.children === l3.children && !re.current) {
              n2 = Ue2(e, n2, t);
              break e;
            }
          } else for (i2 = n2.child, i2 !== null && (i2.return = n2); i2 !== null; ) {
            var o = i2.dependencies;
            if (o !== null) {
              u2 = i2.child;
              for (var s = o.firstContext; s !== null; ) {
                if (s.context === r) {
                  if (i2.tag === 1) {
                    s = Re2(-1, t & -t), s.tag = 2;
                    var d3 = i2.updateQueue;
                    if (d3 !== null) {
                      d3 = d3.shared;
                      var m2 = d3.pending;
                      m2 === null ? s.next = s : (s.next = m2.next, m2.next = s), d3.pending = s;
                    }
                  }
                  i2.lanes |= t, s = i2.alternate, s !== null && (s.lanes |= t), Jl(i2.return, t, n2), o.lanes |= t;
                  break;
                }
                s = s.next;
              }
            } else if (i2.tag === 10) u2 = i2.type === n2.type ? null : i2.child;
            else if (i2.tag === 18) {
              if (u2 = i2.return, u2 === null) throw Error(v2(341));
              u2.lanes |= t, o = u2.alternate, o !== null && (o.lanes |= t), Jl(u2, t, n2), u2 = i2.sibling;
            } else u2 = i2.child;
            if (u2 !== null) u2.return = i2;
            else for (u2 = i2; u2 !== null; ) {
              if (u2 === n2) {
                u2 = null;
                break;
              }
              if (i2 = u2.sibling, i2 !== null) {
                i2.return = u2.return, u2 = i2;
                break;
              }
              u2 = u2.return;
            }
            i2 = u2;
          }
          q2(e, n2, l3.children, t), n2 = n2.child;
        }
        return n2;
      case 9:
        return l3 = n2.type, r = n2.pendingProps.children, An(n2, t), l3 = ve3(l3), r = r(l3), n2.flags |= 1, q2(e, n2, r, t), n2.child;
      case 14:
        return r = n2.type, l3 = we3(r, n2.pendingProps), l3 = we3(r.type, l3), bu(e, n2, r, l3, t);
      case 15:
        return As(e, n2, n2.type, n2.pendingProps, t);
      case 17:
        return r = n2.type, l3 = n2.pendingProps, l3 = n2.elementType === r ? l3 : we3(r, l3), cr(e, n2), n2.tag = 1, le2(r) ? (e = true, xr(n2)) : e = false, An(n2, t), js(n2, r, l3), bl(n2, r, l3, t), ti(null, n2, r, true, e, t);
      case 19:
        return $s(e, n2, t);
      case 22:
        return Bs(e, n2, t);
    }
    throw Error(v2(156, n2.tag));
  };
  function oa(e, n2) {
    return Ro(e, n2);
  }
  function yf(e, n2, t, r) {
    this.tag = e, this.key = t, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.ref = null, this.pendingProps = n2, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = r, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null;
  }
  function me3(e, n2, t, r) {
    return new yf(e, n2, t, r);
  }
  function bi(e) {
    return e = e.prototype, !(!e || !e.isReactComponent);
  }
  function gf(e) {
    if (typeof e == "function") return bi(e) ? 1 : 0;
    if (e != null) {
      if (e = e.$$typeof, e === gi) return 11;
      if (e === wi) return 14;
    }
    return 2;
  }
  function nn(e, n2) {
    var t = e.alternate;
    return t === null ? (t = me3(e.tag, n2, e.key, e.mode), t.elementType = e.elementType, t.type = e.type, t.stateNode = e.stateNode, t.alternate = e, e.alternate = t) : (t.pendingProps = n2, t.type = e.type, t.flags = 0, t.subtreeFlags = 0, t.deletions = null), t.flags = e.flags & 14680064, t.childLanes = e.childLanes, t.lanes = e.lanes, t.child = e.child, t.memoizedProps = e.memoizedProps, t.memoizedState = e.memoizedState, t.updateQueue = e.updateQueue, n2 = e.dependencies, t.dependencies = n2 === null ? null : { lanes: n2.lanes, firstContext: n2.firstContext }, t.sibling = e.sibling, t.index = e.index, t.ref = e.ref, t;
  }
  function pr(e, n2, t, r, l3, i2) {
    var u2 = 2;
    if (r = e, typeof e == "function") bi(e) && (u2 = 1);
    else if (typeof e == "string") u2 = 5;
    else e: switch (e) {
      case Nn:
        return mn(t.children, l3, i2, n2);
      case yi:
        u2 = 8, l3 |= 8;
        break;
      case Cl:
        return e = me3(12, t, n2, l3 | 2), e.elementType = Cl, e.lanes = i2, e;
      case xl:
        return e = me3(13, t, n2, l3), e.elementType = xl, e.lanes = i2, e;
      case Nl:
        return e = me3(19, t, n2, l3), e.elementType = Nl, e.lanes = i2, e;
      case yo:
        return Xr(t, l3, i2, n2);
      default:
        if (typeof e == "object" && e !== null) switch (e.$$typeof) {
          case ho:
            u2 = 10;
            break e;
          case vo:
            u2 = 9;
            break e;
          case gi:
            u2 = 11;
            break e;
          case wi:
            u2 = 14;
            break e;
          case He2:
            u2 = 16, r = null;
            break e;
        }
        throw Error(v2(130, e == null ? e : typeof e, ""));
    }
    return n2 = me3(u2, t, n2, l3), n2.elementType = e, n2.type = r, n2.lanes = i2, n2;
  }
  function mn(e, n2, t, r) {
    return e = me3(7, e, r, n2), e.lanes = t, e;
  }
  function Xr(e, n2, t, r) {
    return e = me3(22, e, r, n2), e.elementType = yo, e.lanes = t, e.stateNode = { isHidden: false }, e;
  }
  function Sl(e, n2, t) {
    return e = me3(6, e, null, n2), e.lanes = t, e;
  }
  function kl(e, n2, t) {
    return n2 = me3(4, e.children !== null ? e.children : [], e.key, n2), n2.lanes = t, n2.stateNode = { containerInfo: e.containerInfo, pendingChildren: null, implementation: e.implementation }, n2;
  }
  function wf(e, n2, t, r, l3) {
    this.tag = n2, this.containerInfo = e, this.finishedWork = this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = -1, this.callbackNode = this.pendingContext = this.context = null, this.callbackPriority = 0, this.eventTimes = ll(0), this.expirationTimes = ll(-1), this.entangledLanes = this.finishedLanes = this.mutableReadLanes = this.expiredLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = ll(0), this.identifierPrefix = r, this.onRecoverableError = l3, this.mutableSourceEagerHydrationData = null;
  }
  function eu(e, n2, t, r, l3, i2, u2, o, s) {
    return e = new wf(e, n2, t, o, s), n2 === 1 ? (n2 = 1, i2 === true && (n2 |= 8)) : n2 = 0, i2 = me3(3, null, null, n2), e.current = i2, i2.stateNode = e, i2.memoizedState = { element: r, isDehydrated: t, cache: null, transitions: null, pendingSuspenseBoundaries: null }, ji(i2), e;
  }
  function Sf(e, n2, t) {
    var r = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
    return { $$typeof: xn, key: r == null ? null : "" + r, children: e, containerInfo: n2, implementation: t };
  }
  function sa(e) {
    if (!e) return rn;
    e = e._reactInternals;
    e: {
      if (kn(e) !== e || e.tag !== 1) throw Error(v2(170));
      var n2 = e;
      do {
        switch (n2.tag) {
          case 3:
            n2 = n2.stateNode.context;
            break e;
          case 1:
            if (le2(n2.type)) {
              n2 = n2.stateNode.__reactInternalMemoizedMergedChildContext;
              break e;
            }
        }
        n2 = n2.return;
      } while (n2 !== null);
      throw Error(v2(171));
    }
    if (e.tag === 1) {
      var t = e.type;
      if (le2(t)) return ss(e, t, n2);
    }
    return n2;
  }
  function aa(e, n2, t, r, l3, i2, u2, o, s) {
    return e = eu(t, r, true, e, l3, i2, u2, o, s), e.context = sa(null), t = e.current, r = b(), l3 = en(t), i2 = Re2(r, l3), i2.callback = n2 ?? null, qe2(t, i2, l3), e.current.lanes = l3, Ft(e, l3, r), ie(e, r), e;
  }
  function Gr(e, n2, t, r) {
    var l3 = n2.current, i2 = b(), u2 = en(l3);
    return t = sa(t), n2.context === null ? n2.context = t : n2.pendingContext = t, n2 = Re2(i2, u2), n2.payload = { element: e }, r = r === void 0 ? null : r, r !== null && (n2.callback = r), e = qe2(l3, n2, u2), e !== null && (Ce2(e, l3, u2, i2), or(e, l3, u2)), u2;
  }
  function jr(e) {
    if (e = e.current, !e.child) return null;
    switch (e.child.tag) {
      case 5:
        return e.child.stateNode;
      default:
        return e.child.stateNode;
    }
  }
  function fo(e, n2) {
    if (e = e.memoizedState, e !== null && e.dehydrated !== null) {
      var t = e.retryLane;
      e.retryLane = t !== 0 && t < n2 ? t : n2;
    }
  }
  function nu(e, n2) {
    fo(e, n2), (e = e.alternate) && fo(e, n2);
  }
  function kf() {
    return null;
  }
  var ca = typeof reportError == "function" ? reportError : function(e) {
    console.error(e);
  };
  function tu(e) {
    this._internalRoot = e;
  }
  Zr.prototype.render = tu.prototype.render = function(e) {
    var n2 = this._internalRoot;
    if (n2 === null) throw Error(v2(409));
    Gr(e, n2, null, null);
  };
  Zr.prototype.unmount = tu.prototype.unmount = function() {
    var e = this._internalRoot;
    if (e !== null) {
      this._internalRoot = null;
      var n2 = e.containerInfo;
      wn(function() {
        Gr(null, e, null, null);
      }), n2[Ie3] = null;
    }
  };
  function Zr(e) {
    this._internalRoot = e;
  }
  Zr.prototype.unstable_scheduleHydration = function(e) {
    if (e) {
      var n2 = Bo();
      e = { blockedOn: null, target: e, priority: n2 };
      for (var t = 0; t < Qe.length && n2 !== 0 && n2 < Qe[t].priority; t++) ;
      Qe.splice(t, 0, e), t === 0 && Wo(e);
    }
  };
  function ru(e) {
    return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11);
  }
  function Jr(e) {
    return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11 && (e.nodeType !== 8 || e.nodeValue !== " react-mount-point-unstable "));
  }
  function po() {
  }
  function Ef(e, n2, t, r, l3) {
    if (l3) {
      if (typeof r == "function") {
        var i2 = r;
        r = function() {
          var d3 = jr(u2);
          i2.call(d3);
        };
      }
      var u2 = aa(n2, r, e, 0, null, false, false, "", po);
      return e._reactRootContainer = u2, e[Ie3] = u2.current, _t(e.nodeType === 8 ? e.parentNode : e), wn(), u2;
    }
    for (; l3 = e.lastChild; ) e.removeChild(l3);
    if (typeof r == "function") {
      var o = r;
      r = function() {
        var d3 = jr(s);
        o.call(d3);
      };
    }
    var s = eu(e, 0, false, null, null, false, false, "", po);
    return e._reactRootContainer = s, e[Ie3] = s.current, _t(e.nodeType === 8 ? e.parentNode : e), wn(function() {
      Gr(n2, s, t, r);
    }), s;
  }
  function qr(e, n2, t, r, l3) {
    var i2 = t._reactRootContainer;
    if (i2) {
      var u2 = i2;
      if (typeof l3 == "function") {
        var o = l3;
        l3 = function() {
          var s = jr(u2);
          o.call(s);
        };
      }
      Gr(n2, u2, e, l3);
    } else u2 = Ef(t, n2, e, l3, r);
    return jr(u2);
  }
  Vo = function(e) {
    switch (e.tag) {
      case 3:
        var n2 = e.stateNode;
        if (n2.current.memoizedState.isDehydrated) {
          var t = ot(n2.pendingLanes);
          t !== 0 && (Ei(n2, t | 1), ie(n2, U3()), (_2 & 6) === 0 && (Xn = U3() + 500, on()));
        }
        break;
      case 13:
        wn(function() {
          var r = je2(e, 1);
          if (r !== null) {
            var l3 = b();
            Ce2(r, e, 1, l3);
          }
        }), nu(e, 1);
    }
  };
  Ci = function(e) {
    if (e.tag === 13) {
      var n2 = je2(e, 134217728);
      if (n2 !== null) {
        var t = b();
        Ce2(n2, e, 134217728, t);
      }
      nu(e, 134217728);
    }
  };
  Ao = function(e) {
    if (e.tag === 13) {
      var n2 = en(e), t = je2(e, n2);
      if (t !== null) {
        var r = b();
        Ce2(t, e, n2, r);
      }
      nu(e, n2);
    }
  };
  Bo = function() {
    return P;
  };
  Ho = function(e, n2) {
    var t = P;
    try {
      return P = e, n2();
    } finally {
      P = t;
    }
  };
  Fl = function(e, n2, t) {
    switch (n2) {
      case "input":
        if (Pl(e, t), n2 = t.name, t.type === "radio" && n2 != null) {
          for (t = e; t.parentNode; ) t = t.parentNode;
          for (t = t.querySelectorAll("input[name=" + JSON.stringify("" + n2) + '][type="radio"]'), n2 = 0; n2 < t.length; n2++) {
            var r = t[n2];
            if (r !== e && r.form === e.form) {
              var l3 = Hr(r);
              if (!l3) throw Error(v2(90));
              wo(r), Pl(r, l3);
            }
          }
        }
        break;
      case "textarea":
        ko(e, t);
        break;
      case "select":
        n2 = t.value, n2 != null && In(e, !!t.multiple, n2, false);
    }
  };
  Po = Zi;
  Lo = wn;
  var Cf = { usingClientEntryPoint: false, Events: [jt, Ln, Hr, _o, zo, Zi] }, rt = { findFiberByHostInstance: cn, bundleType: 0, version: "18.3.1", rendererPackageName: "react-dom" }, xf = { bundleType: rt.bundleType, version: rt.version, rendererPackageName: rt.rendererPackageName, rendererConfig: rt.rendererConfig, overrideHookState: null, overrideHookStateDeletePath: null, overrideHookStateRenamePath: null, overrideProps: null, overridePropsDeletePath: null, overridePropsRenamePath: null, setErrorHandler: null, setSuspenseHandler: null, scheduleUpdate: null, currentDispatcherRef: Ve2.ReactCurrentDispatcher, findHostInstanceByFiber: function(e) {
    return e = Do(e), e === null ? null : e.stateNode;
  }, findFiberByHostInstance: rt.findFiberByHostInstance || kf, findHostInstancesForRefresh: null, scheduleRefresh: null, scheduleRoot: null, setRefreshHandler: null, getCurrentFiber: null, reconcilerVersion: "18.3.1-next-f1338f8080-20240426" };
  if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && (lt = __REACT_DEVTOOLS_GLOBAL_HOOK__, !lt.isDisabled && lt.supportsFiber)) try {
    Ur = lt.inject(xf), Pe3 = lt;
  } catch {
  }
  var lt;
  fe2.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = Cf;
  fe2.createPortal = function(e, n2) {
    var t = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
    if (!ru(n2)) throw Error(v2(200));
    return Sf(e, n2, null, t);
  };
  fe2.createRoot = function(e, n2) {
    if (!ru(e)) throw Error(v2(299));
    var t = false, r = "", l3 = ca;
    return n2 != null && (n2.unstable_strictMode === true && (t = true), n2.identifierPrefix !== void 0 && (r = n2.identifierPrefix), n2.onRecoverableError !== void 0 && (l3 = n2.onRecoverableError)), n2 = eu(e, 1, false, null, null, t, false, r, l3), e[Ie3] = n2.current, _t(e.nodeType === 8 ? e.parentNode : e), new tu(n2);
  };
  fe2.findDOMNode = function(e) {
    if (e == null) return null;
    if (e.nodeType === 1) return e;
    var n2 = e._reactInternals;
    if (n2 === void 0) throw typeof e.render == "function" ? Error(v2(188)) : (e = Object.keys(e).join(","), Error(v2(268, e)));
    return e = Do(n2), e = e === null ? null : e.stateNode, e;
  };
  fe2.flushSync = function(e) {
    return wn(e);
  };
  fe2.hydrate = function(e, n2, t) {
    if (!Jr(n2)) throw Error(v2(200));
    return qr(null, e, n2, true, t);
  };
  fe2.hydrateRoot = function(e, n2, t) {
    if (!ru(e)) throw Error(v2(405));
    var r = t != null && t.hydratedSources || null, l3 = false, i2 = "", u2 = ca;
    if (t != null && (t.unstable_strictMode === true && (l3 = true), t.identifierPrefix !== void 0 && (i2 = t.identifierPrefix), t.onRecoverableError !== void 0 && (u2 = t.onRecoverableError)), n2 = aa(n2, null, e, 1, t ?? null, l3, false, i2, u2), e[Ie3] = n2.current, _t(e), r) for (e = 0; e < r.length; e++) t = r[e], l3 = t._getVersion, l3 = l3(t._source), n2.mutableSourceEagerHydrationData == null ? n2.mutableSourceEagerHydrationData = [t, l3] : n2.mutableSourceEagerHydrationData.push(t, l3);
    return new Zr(n2);
  };
  fe2.render = function(e, n2, t) {
    if (!Jr(n2)) throw Error(v2(200));
    return qr(null, e, n2, false, t);
  };
  fe2.unmountComponentAtNode = function(e) {
    if (!Jr(e)) throw Error(v2(40));
    return e._reactRootContainer ? (wn(function() {
      qr(null, null, e, false, function() {
        e._reactRootContainer = null, e[Ie3] = null;
      });
    }), true) : false;
  };
  fe2.unstable_batchedUpdates = Zi;
  fe2.unstable_renderSubtreeIntoContainer = function(e, n2, t, r) {
    if (!Jr(t)) throw Error(v2(200));
    if (e == null || e._reactInternals === void 0) throw Error(v2(38));
    return qr(e, n2, t, false, r);
  };
  fe2.version = "18.3.1-next-f1338f8080-20240426";
});
var ma = uu((zf, pa) => {
  "use strict";
  function da() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function")) try {
      __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(da);
    } catch (e) {
      console.error(e);
    }
  }
  da(), pa.exports = fa();
});
var br = Ea(ma());
var { __SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED: Pf, createPortal: Lf, createRoot: Tf, findDOMNode: Mf, flushSync: Df, hydrate: Of, hydrateRoot: Rf, render: Ff, unmountComponentAtNode: If, unstable_batchedUpdates: jf, unstable_renderSubtreeIntoContainer: Uf, version: Vf } = br;
var Af = br.default ?? br;

// https://esm.sh/react-dom@18.3.1/denonext/client.mjs
var require3 = (n2) => {
  const e = (m2) => typeof m2.default < "u" ? m2.default : m2, c3 = (m2) => Object.assign({ __esModule: true }, m2);
  switch (n2) {
    case "react-dom":
      return e(react_dom_exports);
    default:
      console.error('module "' + n2 + '" not found');
      return null;
  }
};
var R = Object.create;
var c = Object.defineProperty;
var l = Object.getOwnPropertyDescriptor;
var y = Object.getOwnPropertyNames;
var E = Object.getPrototypeOf;
var _ = Object.prototype.hasOwnProperty;
var f = ((t) => typeof require3 < "u" ? require3 : typeof Proxy < "u" ? new Proxy(t, { get: (o, e) => (typeof require3 < "u" ? require3 : o)[e] }) : t)(function(t) {
  if (typeof require3 < "u") return require3.apply(this, arguments);
  throw Error('Dynamic require of "' + t + '" is not supported');
});
var d = (t, o) => () => (o || t((o = { exports: {} }).exports, o), o.exports);
var m = (t, o, e, a2) => {
  if (o && typeof o == "object" || typeof o == "function") for (let r of y(o)) !_.call(t, r) && r !== e && c(t, r, { get: () => o[r], enumerable: !(a2 = l(o, r)) || a2.enumerable });
  return t;
};
var h2 = (t, o, e) => (e = t != null ? R(E(t)) : {}, m(o || !t || !t.__esModule ? c(e, "default", { value: t, enumerable: true }) : e, t));
var u = d((i2) => {
  "use strict";
  var s = f("react-dom");
  i2.createRoot = s.createRoot, i2.hydrateRoot = s.hydrateRoot;
  var C;
});
var n = h2(u());
var { createRoot: O, hydrateRoot: g } = n;
var x2 = n.default ?? n;

// src/api.ts
var BASE = "http://localhost:8080";
async function newGame() {
  const res = await fetch(`${BASE}/api/games`, { method: "POST" });
  if (!res.ok) throw new Error(`new game failed: ${res.status}`);
  const data = await res.json();
  return data.deck;
}
async function getScores() {
  const res = await fetch(`${BASE}/api/scores`);
  if (!res.ok) throw new Error(`get scores failed: ${res.status}`);
  const data = await res.json();
  return data.scores;
}
async function submitScore(score) {
  const res = await fetch(`${BASE}/api/scores`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(score)
  });
  if (!res.ok) throw new Error(`submit score failed: ${res.status}`);
}

// https://esm.sh/react@18.3.1/denonext/jsx-runtime.mjs
var require4 = (n2) => {
  const e = (m2) => typeof m2.default < "u" ? m2.default : m2, c3 = (m2) => Object.assign({ __esModule: true }, m2);
  switch (n2) {
    case "react":
      return e(react_exports);
    default:
      console.error('module "' + n2 + '" not found');
      return null;
  }
};
var y2 = Object.create;
var l2 = Object.defineProperty;
var j = Object.getOwnPropertyDescriptor;
var x3 = Object.getOwnPropertyNames;
var O2 = Object.getPrototypeOf;
var a = Object.prototype.hasOwnProperty;
var v = ((r) => typeof require4 < "u" ? require4 : typeof Proxy < "u" ? new Proxy(r, { get: (e, o) => (typeof require4 < "u" ? require4 : e)[o] }) : r)(function(r) {
  if (typeof require4 < "u") return require4.apply(this, arguments);
  throw Error('Dynamic require of "' + r + '" is not supported');
});
var i = (r, e) => () => (e || r((e = { exports: {} }).exports, e), e.exports);
var E2 = (r, e, o, t) => {
  if (e && typeof e == "object" || typeof e == "function") for (let s of x3(e)) !a.call(r, s) && s !== o && l2(r, s, { get: () => e[s], enumerable: !(t = j(e, s)) || t.enumerable });
  return r;
};
var k2 = (r, e, o) => (o = r != null ? y2(O2(r)) : {}, E2(e || !r || !r.__esModule ? l2(o, "default", { value: r, enumerable: true }) : o, r));
var c2 = i((n2) => {
  "use strict";
  var N = v("react"), R2 = Symbol.for("react.element"), S2 = Symbol.for("react.fragment"), b = Object.prototype.hasOwnProperty, q2 = N.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner, w2 = { key: true, ref: true, __self: true, __source: true };
  function _2(r, e, o) {
    var t, s = {}, p = null, u2 = null;
    o !== void 0 && (p = "" + o), e.key !== void 0 && (p = "" + e.key), e.ref !== void 0 && (u2 = e.ref);
    for (t in e) b.call(e, t) && !w2.hasOwnProperty(t) && (s[t] = e[t]);
    if (r && r.defaultProps) for (t in e = r.defaultProps, e) s[t] === void 0 && (s[t] = e[t]);
    return { $$typeof: R2, type: r, key: p, ref: u2, props: s, _owner: q2.current };
  }
  n2.Fragment = S2;
  n2.jsx = _2;
  n2.jsxs = _2;
});
var d2 = i((D2, m2) => {
  "use strict";
  m2.exports = c2();
});
var f2 = k2(d2());
var { Fragment: F2, jsx: I, jsxs: L2 } = f2;
var T = f2.default ?? f2;

// src/App.tsx
function toLocalDeck(deck) {
  return deck.map((c3) => ({ ...c3, matched: false }));
}
function App() {
  const [deck, setDeck] = Me([]);
  const [revealed, setRevealed] = Me([]);
  const [moves, setMoves] = Me(0);
  const [locked, setLocked] = Me(false);
  const [startedAt, setStartedAt] = Me(null);
  const [elapsed, setElapsed] = Me(0);
  const [scores, setScores] = Me([]);
  const [name, setName] = Me("");
  const [submitted, setSubmitted] = Me(false);
  const [error, setError] = Me(null);
  const reset = Ie(async () => {
    setError(null);
    try {
      const fresh = await newGame();
      setDeck(toLocalDeck(fresh));
      setRevealed([]);
      setMoves(0);
      setLocked(false);
      setStartedAt(null);
      setElapsed(0);
      setSubmitted(false);
    } catch (e) {
      setError(e.message);
    }
  }, []);
  const refreshScores = Ie(async () => {
    try {
      setScores(await getScores());
    } catch (e) {
      setError(e.message);
    }
  }, []);
  De(() => {
    reset();
    refreshScores();
  }, [reset, refreshScores]);
  const won = Ue(
    () => deck.length > 0 && deck.every((c3) => c3.matched),
    [deck]
  );
  De(() => {
    if (won || startedAt === null) return;
    const t = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAt) / 1e3));
    }, 250);
    return () => clearInterval(t);
  }, [won, startedAt]);
  De(() => {
    if (revealed.length !== 2) return;
    setLocked(true);
    const [a2, b] = revealed;
    const cardA = deck.find((c3) => c3.id === a2);
    const cardB = deck.find((c3) => c3.id === b);
    if (cardA.symbol === cardB.symbol) {
      setDeck(
        (d3) => d3.map((c3) => c3.id === a2 || c3.id === b ? { ...c3, matched: true } : c3)
      );
      setRevealed([]);
      setLocked(false);
    } else {
      const timer = setTimeout(() => {
        setRevealed([]);
        setLocked(false);
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [revealed, deck]);
  function handleClick(card) {
    if (locked || card.matched || revealed.includes(card.id)) return;
    if (startedAt === null) setStartedAt(Date.now());
    if (revealed.length === 0) {
      setRevealed([card.id]);
      setMoves((m2) => m2 + 1);
    } else if (revealed.length === 1) {
      setRevealed([revealed[0], card.id]);
    }
  }
  async function handleSubmit() {
    if (!name.trim() || submitted || !won) return;
    try {
      await submitScore({ name: name.trim(), moves, seconds: Math.max(1, elapsed) });
      setSubmitted(true);
      await refreshScores();
    } catch (e) {
      setError(e.message);
    }
  }
  const matchedCount = deck.filter((c3) => c3.matched).length / 2;
  return /* @__PURE__ */ L2("div", { className: "game", children: [
    /* @__PURE__ */ L2("header", { className: "header", children: [
      /* @__PURE__ */ I("h1", { children: "Memory Game" }),
      /* @__PURE__ */ L2("div", { className: "stats", children: [
        /* @__PURE__ */ L2("span", { children: [
          "Moves: ",
          moves
        ] }),
        /* @__PURE__ */ L2("span", { children: [
          "Time: ",
          elapsed,
          "s"
        ] }),
        /* @__PURE__ */ L2("span", { children: [
          "Pairs: ",
          matchedCount,
          " / ",
          deck.length / 2 || 0
        ] })
      ] })
    ] }),
    /* @__PURE__ */ I("div", { className: "controls", children: /* @__PURE__ */ I("button", { onClick: reset, children: "New Game" }) }),
    error && /* @__PURE__ */ I("div", { className: "error", children: error }),
    won && !submitted && /* @__PURE__ */ L2("div", { className: "win", children: [
      /* @__PURE__ */ L2("p", { children: [
        "You won in ",
        moves,
        " moves and ",
        elapsed,
        "s!"
      ] }),
      /* @__PURE__ */ L2("div", { className: "submit-row", children: [
        /* @__PURE__ */ I(
          "input",
          {
            type: "text",
            placeholder: "Your name",
            value: name,
            onChange: (e) => setName(e.target.value),
            maxLength: 24
          }
        ),
        /* @__PURE__ */ I("button", { onClick: handleSubmit, disabled: !name.trim(), children: "Save Score" })
      ] })
    ] }),
    won && submitted && /* @__PURE__ */ I("div", { className: "win", children: "Score saved." }),
    /* @__PURE__ */ I("div", { className: "board", children: deck.map((card) => {
      const isFlipped = card.matched || revealed.includes(card.id);
      const classes = [
        "card",
        isFlipped ? "flipped" : "",
        card.matched ? "matched" : ""
      ].filter(Boolean).join(" ");
      return /* @__PURE__ */ I(
        "div",
        {
          className: classes,
          onClick: () => handleClick(card),
          children: /* @__PURE__ */ L2("div", { className: "card-inner", children: [
            /* @__PURE__ */ I("div", { className: "card-face card-front", children: "?" }),
            /* @__PURE__ */ I("div", { className: "card-face card-back", children: card.symbol })
          ] })
        },
        card.id
      );
    }) }),
    /* @__PURE__ */ L2("section", { className: "leaderboard", children: [
      /* @__PURE__ */ I("h2", { children: "Leaderboard" }),
      scores.length === 0 && /* @__PURE__ */ I("p", { className: "empty", children: "No scores yet." }),
      scores.length > 0 && /* @__PURE__ */ I("ol", { children: scores.map((s, i2) => /* @__PURE__ */ L2("li", { children: [
        /* @__PURE__ */ I("span", { children: s.name }),
        /* @__PURE__ */ L2("span", { children: [
          s.moves,
          " moves \xB7 ",
          s.seconds,
          "s"
        ] })
      ] }, i2)) })
    ] })
  ] });
}

// src/main.tsx
var container = document.getElementById("root");
if (!container) throw new Error("root container missing");
O(container).render(/* @__PURE__ */ I(App, {}));
/*! Bundled license information:

react/cjs/react.production.min.js:
  (**
   * @license React
   * react.production.min.js
   *
   * Copyright (c) Facebook, Inc. and its affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)
*/
/*! Bundled license information:

scheduler/cjs/scheduler.production.min.js:
  (**
   * @license React
   * scheduler.production.min.js
   *
   * Copyright (c) Facebook, Inc. and its affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)
*/
/*! Bundled license information:

react-dom/cjs/react-dom.production.min.js:
  (**
   * @license React
   * react-dom.production.min.js
   *
   * Copyright (c) Facebook, Inc. and its affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)
*/
/*! Bundled license information:

react/cjs/react-jsx-runtime.production.min.js:
  (**
   * @license React
   * react-jsx-runtime.production.min.js
   *
   * Copyright (c) Facebook, Inc. and its affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)
*/
//# sourceMappingURL=bundle.js.map

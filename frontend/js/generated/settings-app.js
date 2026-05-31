var DocFlowSettingsApp = (function(exports) {
	Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
	//#region node_modules/preact/dist/preact.module.js
	var n, l, u$1, i$1, r, o$1, e, f$1, c, a, s, h, p, v, d = {}, w = [], _ = /acit|ex(?:s|g|n|p|$)|rph|grid|ows|mnc|ntw|ine[ch]|zoo|^ord|itera/i, g = Array.isArray;
	function m(n, l) {
		for (var u in l) n[u] = l[u];
		return n;
	}
	function b(n) {
		n && n.parentNode && n.parentNode.removeChild(n);
	}
	function k(l, u, t) {
		var i, r, o, e = {};
		for (o in u) "key" == o ? i = u[o] : "ref" == o ? r = u[o] : e[o] = u[o];
		if (arguments.length > 2 && (e.children = arguments.length > 3 ? n.call(arguments, 2) : t), "function" == typeof l && null != l.defaultProps) for (o in l.defaultProps) void 0 === e[o] && (e[o] = l.defaultProps[o]);
		return x(l, e, i, r, null);
	}
	function x(n, t, i, r, o) {
		var e = {
			type: n,
			props: t,
			key: i,
			ref: r,
			__k: null,
			__: null,
			__b: 0,
			__e: null,
			__c: null,
			constructor: void 0,
			__v: null == o ? ++u$1 : o,
			__i: -1,
			__u: 0
		};
		return null == o && null != l.vnode && l.vnode(e), e;
	}
	function S(n) {
		return n.children;
	}
	function C(n, l) {
		this.props = n, this.context = l;
	}
	function $(n, l) {
		if (null == l) return n.__ ? $(n.__, n.__i + 1) : null;
		for (var u; l < n.__k.length; l++) if (null != (u = n.__k[l]) && null != u.__e) return u.__e;
		return "function" == typeof n.type ? $(n) : null;
	}
	function I(n) {
		if (n.__P && n.__d) {
			var u = n.__v, t = u.__e, i = [], r = [], o = m({}, u);
			o.__v = u.__v + 1, l.vnode && l.vnode(o), q(n.__P, o, u, n.__n, n.__P.namespaceURI, 32 & u.__u ? [t] : null, i, null == t ? $(u) : t, !!(32 & u.__u), r), o.__v = u.__v, o.__.__k[o.__i] = o, D(i, o, r), u.__e = u.__ = null, o.__e != t && P(o);
		}
	}
	function P(n) {
		if (null != (n = n.__) && null != n.__c) return n.__e = n.__c.base = null, n.__k.some(function(l) {
			if (null != l && null != l.__e) return n.__e = n.__c.base = l.__e;
		}), P(n);
	}
	function A(n) {
		(!n.__d && (n.__d = !0) && i$1.push(n) && !H.__r++ || r != l.debounceRendering) && ((r = l.debounceRendering) || o$1)(H);
	}
	function H() {
		try {
			for (var n, l = 1; i$1.length;) i$1.length > l && i$1.sort(e), n = i$1.shift(), l = i$1.length, I(n);
		} finally {
			i$1.length = H.__r = 0;
		}
	}
	function L(n, l, u, t, i, r, o, e, f, c, a) {
		var s, h, p, v, y, _, g, m = t && t.__k || w, b = l.length;
		for (f = T(u, l, m, f, b), s = 0; s < b; s++) null != (p = u.__k[s]) && (h = -1 != p.__i && m[p.__i] || d, p.__i = s, _ = q(n, p, h, i, r, o, e, f, c, a), v = p.__e, p.ref && h.ref != p.ref && (h.ref && J(h.ref, null, p), a.push(p.ref, p.__c || v, p)), null == y && null != v && (y = v), (g = !!(4 & p.__u)) || h.__k === p.__k ? (f = j(p, f, n, g), g && h.__e && (h.__e = null)) : "function" == typeof p.type && void 0 !== _ ? f = _ : v && (f = v.nextSibling), p.__u &= -7);
		return u.__e = y, f;
	}
	function T(n, l, u, t, i) {
		var r, o, e, f, c, a = u.length, s = a, h = 0;
		for (n.__k = new Array(i), r = 0; r < i; r++) null != (o = l[r]) && "boolean" != typeof o && "function" != typeof o ? ("string" == typeof o || "number" == typeof o || "bigint" == typeof o || o.constructor == String ? o = n.__k[r] = x(null, o, null, null, null) : g(o) ? o = n.__k[r] = x(S, { children: o }, null, null, null) : void 0 === o.constructor && o.__b > 0 ? o = n.__k[r] = x(o.type, o.props, o.key, o.ref ? o.ref : null, o.__v) : n.__k[r] = o, f = r + h, o.__ = n, o.__b = n.__b + 1, e = null, -1 != (c = o.__i = O(o, u, f, s)) && (s--, (e = u[c]) && (e.__u |= 2)), null == e || null == e.__v ? (-1 == c && (i > a ? h-- : i < a && h++), "function" != typeof o.type && (o.__u |= 4)) : c != f && (c == f - 1 ? h-- : c == f + 1 ? h++ : (c > f ? h-- : h++, o.__u |= 4))) : n.__k[r] = null;
		if (s) for (r = 0; r < a; r++) null != (e = u[r]) && 0 == (2 & e.__u) && (e.__e == t && (t = $(e)), K(e, e));
		return t;
	}
	function j(n, l, u, t) {
		var i, r;
		if ("function" == typeof n.type) {
			for (i = n.__k, r = 0; i && r < i.length; r++) i[r] && (i[r].__ = n, l = j(i[r], l, u, t));
			return l;
		}
		n.__e != l && (t && (l && n.type && !l.parentNode && (l = $(n)), u.insertBefore(n.__e, l || null)), l = n.__e);
		do
			l = l && l.nextSibling;
		while (null != l && 8 == l.nodeType);
		return l;
	}
	function O(n, l, u, t) {
		var i, r, o, e = n.key, f = n.type, c = l[u], a = null != c && 0 == (2 & c.__u);
		if (null === c && null == e || a && e == c.key && f == c.type) return u;
		if (t > (a ? 1 : 0)) {
			for (i = u - 1, r = u + 1; i >= 0 || r < l.length;) if (null != (c = l[o = i >= 0 ? i-- : r++]) && 0 == (2 & c.__u) && e == c.key && f == c.type) return o;
		}
		return -1;
	}
	function z(n, l, u) {
		"-" == l[0] ? n.setProperty(l, null == u ? "" : u) : n[l] = null == u ? "" : "number" != typeof u || _.test(l) ? u : u + "px";
	}
	function N(n, l, u, t, i) {
		var r, o;
		n: if ("style" == l) if ("string" == typeof u) n.style.cssText = u;
		else {
			if ("string" == typeof t && (n.style.cssText = t = ""), t) for (l in t) u && l in u || z(n.style, l, "");
			if (u) for (l in u) t && u[l] == t[l] || z(n.style, l, u[l]);
		}
		else if ("o" == l[0] && "n" == l[1]) r = l != (l = l.replace(s, "$1")), o = l.toLowerCase(), l = o in n || "onFocusOut" == l || "onFocusIn" == l ? o.slice(2) : l.slice(2), n.l || (n.l = {}), n.l[l + r] = u, u ? t ? u[a] = t[a] : (u[a] = h, n.addEventListener(l, r ? v : p, r)) : n.removeEventListener(l, r ? v : p, r);
		else {
			if ("http://www.w3.org/2000/svg" == i) l = l.replace(/xlink(H|:h)/, "h").replace(/sName$/, "s");
			else if ("width" != l && "height" != l && "href" != l && "list" != l && "form" != l && "tabIndex" != l && "download" != l && "rowSpan" != l && "colSpan" != l && "role" != l && "popover" != l && l in n) try {
				n[l] = null == u ? "" : u;
				break n;
			} catch (n) {}
			"function" == typeof u || (null == u || !1 === u && "-" != l[4] ? n.removeAttribute(l) : n.setAttribute(l, "popover" == l && 1 == u ? "" : u));
		}
	}
	function V(n) {
		return function(u) {
			if (this.l) {
				var t = this.l[u.type + n];
				if (null == u[c]) u[c] = h++;
				else if (u[c] < t[a]) return;
				return t(l.event ? l.event(u) : u);
			}
		};
	}
	function q(n, u, t, i, r, o, e, f, c, a) {
		var s, h, p, v, y, d, _, k, x, M, $, I, P, A, H, T = u.type;
		if (void 0 !== u.constructor) return null;
		128 & t.__u && (c = !!(32 & t.__u), o = [f = u.__e = t.__e]), (s = l.__b) && s(u);
		n: if ("function" == typeof T) try {
			if (k = u.props, x = T.prototype && T.prototype.render, M = (s = T.contextType) && i[s.__c], $ = s ? M ? M.props.value : s.__ : i, t.__c ? _ = (h = u.__c = t.__c).__ = h.__E : (x ? u.__c = h = new T(k, $) : (u.__c = h = new C(k, $), h.constructor = T, h.render = Q), M && M.sub(h), h.state || (h.state = {}), h.__n = i, p = h.__d = !0, h.__h = [], h._sb = []), x && null == h.__s && (h.__s = h.state), x && null != T.getDerivedStateFromProps && (h.__s == h.state && (h.__s = m({}, h.__s)), m(h.__s, T.getDerivedStateFromProps(k, h.__s))), v = h.props, y = h.state, h.__v = u, p) x && null == T.getDerivedStateFromProps && null != h.componentWillMount && h.componentWillMount(), x && null != h.componentDidMount && h.__h.push(h.componentDidMount);
			else {
				if (x && null == T.getDerivedStateFromProps && k !== v && null != h.componentWillReceiveProps && h.componentWillReceiveProps(k, $), u.__v == t.__v || !h.__e && null != h.shouldComponentUpdate && !1 === h.shouldComponentUpdate(k, h.__s, $)) {
					u.__v != t.__v && (h.props = k, h.state = h.__s, h.__d = !1), u.__e = t.__e, u.__k = t.__k, u.__k.some(function(n) {
						n && (n.__ = u);
					}), w.push.apply(h.__h, h._sb), h._sb = [], h.__h.length && e.push(h);
					break n;
				}
				null != h.componentWillUpdate && h.componentWillUpdate(k, h.__s, $), x && null != h.componentDidUpdate && h.__h.push(function() {
					h.componentDidUpdate(v, y, d);
				});
			}
			if (h.context = $, h.props = k, h.__P = n, h.__e = !1, I = l.__r, P = 0, x) h.state = h.__s, h.__d = !1, I && I(u), s = h.render(h.props, h.state, h.context), w.push.apply(h.__h, h._sb), h._sb = [];
			else do
				h.__d = !1, I && I(u), s = h.render(h.props, h.state, h.context), h.state = h.__s;
			while (h.__d && ++P < 25);
			h.state = h.__s, null != h.getChildContext && (i = m(m({}, i), h.getChildContext())), x && !p && null != h.getSnapshotBeforeUpdate && (d = h.getSnapshotBeforeUpdate(v, y)), A = null != s && s.type === S && null == s.key ? E(s.props.children) : s, f = L(n, g(A) ? A : [A], u, t, i, r, o, e, f, c, a), h.base = u.__e, u.__u &= -161, h.__h.length && e.push(h), _ && (h.__E = h.__ = null);
		} catch (n) {
			if (u.__v = null, c || null != o) if (n.then) {
				for (u.__u |= c ? 160 : 128; f && 8 == f.nodeType && f.nextSibling;) f = f.nextSibling;
				o[o.indexOf(f)] = null, u.__e = f;
			} else {
				for (H = o.length; H--;) b(o[H]);
				B(u);
			}
			else u.__e = t.__e, u.__k = t.__k, n.then || B(u);
			l.__e(n, u, t);
		}
		else null == o && u.__v == t.__v ? (u.__k = t.__k, u.__e = t.__e) : f = u.__e = G(t.__e, u, t, i, r, o, e, c, a);
		return (s = l.diffed) && s(u), 128 & u.__u ? void 0 : f;
	}
	function B(n) {
		n && (n.__c && (n.__c.__e = !0), n.__k && n.__k.some(B));
	}
	function D(n, u, t) {
		for (var i = 0; i < t.length; i++) J(t[i], t[++i], t[++i]);
		l.__c && l.__c(u, n), n.some(function(u) {
			try {
				n = u.__h, u.__h = [], n.some(function(n) {
					n.call(u);
				});
			} catch (n) {
				l.__e(n, u.__v);
			}
		});
	}
	function E(n) {
		return "object" != typeof n || null == n || n.__b > 0 ? n : g(n) ? n.map(E) : void 0 !== n.constructor ? null : m({}, n);
	}
	function G(u, t, i, r, o, e, f, c, a) {
		var s, h, p, v, y, w, _, m = i.props || d, k = t.props, x = t.type;
		if ("svg" == x ? o = "http://www.w3.org/2000/svg" : "math" == x ? o = "http://www.w3.org/1998/Math/MathML" : o || (o = "http://www.w3.org/1999/xhtml"), null != e) {
			for (s = 0; s < e.length; s++) if ((y = e[s]) && "setAttribute" in y == !!x && (x ? y.localName == x : 3 == y.nodeType)) {
				u = y, e[s] = null;
				break;
			}
		}
		if (null == u) {
			if (null == x) return document.createTextNode(k);
			u = document.createElementNS(o, x, k.is && k), c && (l.__m && l.__m(t, e), c = !1), e = null;
		}
		if (null == x) m === k || c && u.data == k || (u.data = k);
		else {
			if (e = "textarea" == x && null != k.defaultValue ? null : e && n.call(u.childNodes), !c && null != e) for (m = {}, s = 0; s < u.attributes.length; s++) m[(y = u.attributes[s]).name] = y.value;
			for (s in m) y = m[s], "dangerouslySetInnerHTML" == s ? p = y : "children" == s || s in k || "value" == s && "defaultValue" in k || "checked" == s && "defaultChecked" in k || N(u, s, null, y, o);
			for (s in k) y = k[s], "children" == s ? v = y : "dangerouslySetInnerHTML" == s ? h = y : "value" == s ? w = y : "checked" == s ? _ = y : c && "function" != typeof y || m[s] === y || N(u, s, y, m[s], o);
			if (h) c || p && (h.__html == p.__html || h.__html == u.innerHTML) || (u.innerHTML = h.__html), t.__k = [];
			else if (p && (u.innerHTML = ""), L("template" == t.type ? u.content : u, g(v) ? v : [v], t, i, r, "foreignObject" == x ? "http://www.w3.org/1999/xhtml" : o, e, f, e ? e[0] : i.__k && $(i, 0), c, a), null != e) for (s = e.length; s--;) b(e[s]);
			c && "textarea" != x || (s = "value", "progress" == x && null == w ? u.removeAttribute("value") : null != w && (w !== u[s] || "progress" == x && !w || "option" == x && w != m[s]) && N(u, s, w, m[s], o), s = "checked", null != _ && _ != u[s] && N(u, s, _, m[s], o));
		}
		return u;
	}
	function J(n, u, t) {
		try {
			if ("function" == typeof n) {
				var i = "function" == typeof n.__u;
				i && n.__u(), i && null == u || (n.__u = n(u));
			} else n.current = u;
		} catch (n) {
			l.__e(n, t);
		}
	}
	function K(n, u, t) {
		var i, r;
		if (l.unmount && l.unmount(n), (i = n.ref) && (i.current && i.current != n.__e || J(i, null, u)), null != (i = n.__c)) {
			if (i.componentWillUnmount) try {
				i.componentWillUnmount();
			} catch (n) {
				l.__e(n, u);
			}
			i.base = i.__P = null;
		}
		if (i = n.__k) for (r = 0; r < i.length; r++) i[r] && K(i[r], u, t || "function" != typeof n.type);
		t || b(n.__e), n.__c = n.__ = n.__e = void 0;
	}
	function Q(n, l, u) {
		return this.constructor(n, u);
	}
	function R(u, t, i) {
		var r, o, e, f;
		t == document && (t = document.documentElement), l.__ && l.__(u, t), o = (r = "function" == typeof i) ? null : i && i.__k || t.__k, e = [], f = [], q(t, u = (!r && i || t).__k = k(S, null, [u]), o || d, d, t.namespaceURI, !r && i ? [i] : o ? null : t.firstChild ? n.call(t.childNodes) : null, e, !r && i ? i : o ? o.__e : t.firstChild, r, f), D(e, u, f);
	}
	n = w.slice, l = { __e: function(n, l, u, t) {
		for (var i, r, o; l = l.__;) if ((i = l.__c) && !i.__) try {
			if ((r = i.constructor) && null != r.getDerivedStateFromError && (i.setState(r.getDerivedStateFromError(n)), o = i.__d), null != i.componentDidCatch && (i.componentDidCatch(n, t || {}), o = i.__d), o) return i.__E = i;
		} catch (l) {
			n = l;
		}
		throw n;
	} }, u$1 = 0, C.prototype.setState = function(n, l) {
		var u = null != this.__s && this.__s != this.state ? this.__s : this.__s = m({}, this.state);
		"function" == typeof n && (n = n(m({}, u), this.props)), n && m(u, n), null != n && this.__v && (l && this._sb.push(l), A(this));
	}, C.prototype.forceUpdate = function(n) {
		this.__v && (this.__e = !0, n && this.__h.push(n), A(this));
	}, C.prototype.render = S, i$1 = [], o$1 = "function" == typeof Promise ? Promise.prototype.then.bind(Promise.resolve()) : setTimeout, e = function(n, l) {
		return n.__v.__b - l.__v.__b;
	}, H.__r = 0, f$1 = Math.random().toString(8), c = "__d" + f$1, a = "__a" + f$1, s = /(PointerCapture)$|Capture$/i, h = 0, p = V(!1), v = V(!0);
	//#endregion
	//#region frontend/src/design-system.ts
	var iconSizes = {
		xs: "14px",
		sm: "15px",
		md: "16px",
		lg: "18px"
	};
	var panelClass = "rounded-xl bg-surface-container-lowest shadow-sm p-5";
	var mutedCardClass = "rounded-lg bg-surface-container-low px-3 py-3";
	var settingsViewContract = {
		rootId: "settings-view-root",
		viewId: "view-settings",
		requiredIds: [
			"settings-title",
			"health-icon",
			"health-label",
			"health-details",
			"settings-sources-list",
			"llm-status",
			"llm-btn",
			"llm-current",
			"llm-dropdown",
			"settings-model-list",
			"theme-toggle-btn",
			"settings-insights-list",
			"settings-storage-list"
		]
	}, f = 0;
	Array.isArray;
	function u(e, t, n, o, i, u) {
		t || (t = {});
		var a, c, p = t;
		if ("ref" in p) for (c in p = {}, t) "ref" == c ? a = t[c] : p[c] = t[c];
		var l$1 = {
			type: e,
			props: p,
			key: n,
			ref: a,
			__k: null,
			__: null,
			__b: 0,
			__e: null,
			__c: null,
			constructor: void 0,
			__v: --f,
			__i: -1,
			__u: 0,
			__source: i,
			__self: u
		};
		if ("function" == typeof e && (a = e.defaultProps)) for (c in a) void 0 === p[c] && (p[c] = a[c]);
		return l.vnode && l.vnode(l$1), l$1;
	}
	//#endregion
	//#region frontend/src/settings-app.tsx
	var win = window;
	function call(action, ...args) {
		return (event) => {
			if (action === "toggleHealthPanel") {
				win.toggleHealthPanel?.(event);
				return;
			}
			const fn = win[action];
			if (typeof fn === "function") fn(...args);
		};
	}
	function Icon({ name, size = iconSizes.sm, className = "" }) {
		return /* @__PURE__ */ u("span", {
			class: `material-symbols-outlined ${className}`,
			style: { fontSize: size },
			children: name
		});
	}
	function SettingsHeader() {
		return /* @__PURE__ */ u("header", {
			class: "h-16 flex-shrink-0 flex justify-between items-center px-8 bg-surface-container-lowest/90 z-40",
			children: [/* @__PURE__ */ u("div", { children: [/* @__PURE__ */ u("h1", {
				id: "settings-title",
				class: "text-base font-semibold text-on-surface tracking-tight",
				"data-i18n": "nav.settings",
				children: "设置"
			}), /* @__PURE__ */ u("p", {
				class: "text-[11px] text-on-surface-variant/60 mt-0.5",
				"data-i18n": "settings.subtitle",
				children: "本地状态、模型和资料来源"
			})] }), /* @__PURE__ */ u("button", {
				onClick: call("refreshSettings"),
				title: "刷新设置状态",
				"aria-label": "刷新设置状态",
				class: "toolbar-btn whitespace-nowrap",
				children: [/* @__PURE__ */ u(Icon, {
					name: "sync",
					size: iconSizes.md
				}), /* @__PURE__ */ u("span", {
					class: "hidden sm:inline",
					children: "刷新状态"
				})]
			})]
		});
	}
	function HealthPanel() {
		return /* @__PURE__ */ u("section", {
			class: panelClass,
			children: [/* @__PURE__ */ u("div", {
				class: "flex items-center justify-between gap-3 mb-4",
				children: [/* @__PURE__ */ u("div", {
					class: "flex items-center gap-2",
					children: [/* @__PURE__ */ u("span", {
						id: "health-icon",
						class: "w-2 h-2 rounded-full bg-tertiary"
					}), /* @__PURE__ */ u("h2", {
						class: "text-sm font-semibold text-on-surface",
						children: "系统状态"
					})]
				}), /* @__PURE__ */ u("button", {
					id: "health-btn",
					onClick: call("toggleHealthPanel"),
					class: "flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-container text-xs font-medium text-on-surface-variant hover:bg-surface-container-high active:scale-95 transition-all",
					children: [/* @__PURE__ */ u("span", {
						id: "health-label",
						children: "状态"
					}), /* @__PURE__ */ u(Icon, {
						name: "refresh",
						size: iconSizes.xs
					})]
				})]
			}), /* @__PURE__ */ u("div", {
				id: "health-panel",
				class: "max-h-[300px] overflow-y-auto custom-scrollbar rounded-lg bg-surface-container-low p-3",
				children: /* @__PURE__ */ u("div", {
					id: "health-details",
					class: "grid grid-cols-2 xl:grid-cols-3 gap-2 text-xs text-on-surface-variant"
				})
			})]
		});
	}
	function SourcesPanel() {
		return /* @__PURE__ */ u("section", {
			class: panelClass,
			children: [/* @__PURE__ */ u("div", {
				class: "flex items-center gap-2 mb-4",
				children: [/* @__PURE__ */ u(Icon, {
					name: "folder_managed",
					size: iconSizes.lg,
					className: "text-primary"
				}), /* @__PURE__ */ u("h2", {
					class: "text-sm font-semibold text-on-surface",
					children: "监控目录"
				})]
			}), /* @__PURE__ */ u("div", {
				id: "settings-sources-list",
				class: "flex flex-col gap-2 text-xs text-on-surface-variant"
			})]
		});
	}
	function ModelPanel() {
		return /* @__PURE__ */ u("section", {
			class: panelClass,
			children: [/* @__PURE__ */ u("div", {
				class: "flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4",
				children: [/* @__PURE__ */ u("div", { children: [/* @__PURE__ */ u("h2", {
					class: "text-sm font-semibold text-on-surface",
					children: "本地模型"
				}), /* @__PURE__ */ u("p", {
					id: "llm-status",
					class: "text-[11px] text-on-surface-variant/60 mt-0.5",
					children: "本地模型"
				})] }), /* @__PURE__ */ u("div", {
					class: "relative",
					children: [/* @__PURE__ */ u("button", {
						id: "llm-btn",
						class: "flex w-full sm:w-auto items-center justify-between sm:justify-start gap-2 px-3 py-1.5 rounded-lg bg-surface-container text-xs font-medium text-on-surface-variant hover:bg-surface-container-high active:scale-95 transition-all",
						children: [
							/* @__PURE__ */ u(Icon, {
								name: "bolt",
								size: iconSizes.xs,
								className: "text-primary"
							}),
							/* @__PURE__ */ u("span", {
								id: "llm-current",
								class: "line-clamp-2 text-left",
								children: "读取中"
							}),
							/* @__PURE__ */ u(Icon, {
								name: "keyboard_arrow_down",
								size: iconSizes.xs,
								className: "text-on-surface-variant"
							})
						]
					}), /* @__PURE__ */ u("div", {
						id: "llm-dropdown",
						class: "hidden absolute right-0 mt-1 bg-surface-container-lowest shadow-xl rounded-xl overflow-hidden z-50 min-w-max"
					})]
				})]
			}), /* @__PURE__ */ u("div", {
				id: "settings-model-list",
				class: "flex flex-col gap-2 text-xs text-on-surface-variant"
			})]
		});
	}
	function PreferencePanel() {
		return /* @__PURE__ */ u("section", {
			class: panelClass,
			children: [
				/* @__PURE__ */ u("div", {
					class: "flex items-center gap-2 mb-4",
					children: [/* @__PURE__ */ u(Icon, {
						name: "tune",
						size: iconSizes.lg,
						className: "text-primary"
					}), /* @__PURE__ */ u("h2", {
						class: "text-sm font-semibold text-on-surface",
						children: "使用偏好"
					})]
				}),
				/* @__PURE__ */ u("div", {
					class: "grid grid-cols-1 sm:grid-cols-3 gap-2 mb-4",
					children: [
						/* @__PURE__ */ u("button", {
							onClick: call("triggerIngest"),
							class: "toolbar-btn toolbar-btn-primary justify-center",
							children: [/* @__PURE__ */ u(Icon, { name: "folder_sync" }), "扫描文件夹"]
						}),
						/* @__PURE__ */ u("button", {
							onClick: call("switchView", "history"),
							class: "toolbar-btn justify-center",
							children: [/* @__PURE__ */ u(Icon, { name: "history" }), "历史记录"]
						}),
						/* @__PURE__ */ u("button", {
							onClick: call("switchView", "library"),
							class: "toolbar-btn justify-center",
							children: [/* @__PURE__ */ u(Icon, { name: "folder_open" }), "文件库"]
						})
					]
				}),
				/* @__PURE__ */ u(PreferenceRow, {
					title: "界面语言",
					detail: "切换常用导航和状态文案。",
					labelId: "locale-current-label",
					label: "中文",
					icon: "language",
					onClick: call("toggleLocale"),
					ariaLabel: "切换语言",
					titleKey: "locale.label",
					detailKey: "locale.detail",
					ariaKey: "locale.toggle"
				}),
				/* @__PURE__ */ u(PreferenceRow, {
					title: "界面主题",
					detail: "浅色和深色外观会保持相同的信息层级。",
					labelId: "theme-current-label",
					label: "浅色",
					icon: "lightbulb",
					onClick: call("toggleTheme"),
					buttonId: "theme-toggle-btn",
					ariaLabel: "切换主题",
					titleKey: "theme.label",
					detailKey: "theme.detail",
					ariaKey: "theme.toggle"
				}),
				/* @__PURE__ */ u("div", {
					class: "grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs",
					children: [/* @__PURE__ */ u("div", {
						class: mutedCardClass,
						children: [/* @__PURE__ */ u("div", {
							class: "font-semibold text-on-surface",
							children: "默认本地优先"
						}), /* @__PURE__ */ u("div", {
							class: "mt-1 text-[11px] leading-relaxed text-on-surface-variant/70",
							children: "资料、整理结果和历史都保存在本机。"
						})]
					}), /* @__PURE__ */ u("div", {
						class: mutedCardClass,
						children: [/* @__PURE__ */ u("div", {
							class: "font-semibold text-on-surface",
							children: "手动整理"
						}), /* @__PURE__ */ u("div", {
							class: "mt-1 text-[11px] leading-relaxed text-on-surface-variant/70",
							children: "从文件库统一管理收藏、集合和标签。"
						})]
					})]
				})
			]
		});
	}
	function PreferenceRow(props) {
		return /* @__PURE__ */ u("div", {
			class: `${mutedCardClass} mb-3`,
			children: /* @__PURE__ */ u("div", {
				class: "flex items-center justify-between gap-3",
				children: [/* @__PURE__ */ u("div", { children: [/* @__PURE__ */ u("div", {
					class: "font-semibold text-on-surface",
					"data-i18n": props.titleKey,
					children: props.title
				}), /* @__PURE__ */ u("div", {
					class: "mt-1 text-[11px] leading-relaxed text-on-surface-variant/70",
					"data-i18n": props.detailKey,
					children: props.detail
				})] }), /* @__PURE__ */ u("button", {
					id: props.buttonId,
					onClick: props.onClick,
					class: "toolbar-btn justify-center",
					"aria-label": props.ariaLabel,
					"data-i18n-aria": props.ariaKey,
					children: [/* @__PURE__ */ u(Icon, { name: props.icon }), /* @__PURE__ */ u("span", {
						id: props.labelId,
						children: props.label
					})]
				})]
			})
		});
	}
	function ContextPanel() {
		return /* @__PURE__ */ u("aside", {
			class: "context-panel overflow-y-auto custom-scrollbar p-4",
			children: [
				/* @__PURE__ */ u("section", {
					class: "soft-panel p-4",
					children: [/* @__PURE__ */ u("div", {
						class: "flex items-start justify-between gap-3",
						children: [/* @__PURE__ */ u("div", { children: [/* @__PURE__ */ u("h2", {
							class: "panel-title",
							children: "状态提示"
						}), /* @__PURE__ */ u("p", {
							class: "panel-muted mt-1",
							children: "基于当前本地服务状态生成。"
						})] }), /* @__PURE__ */ u("button", {
							onClick: call("refreshSettings"),
							class: "icon-button !w-8 !h-8",
							title: "刷新状态提示",
							"aria-label": "刷新状态提示",
							children: /* @__PURE__ */ u(Icon, { name: "sync" })
						})]
					}), /* @__PURE__ */ u("div", {
						id: "settings-insights-list",
						role: "status",
						"aria-live": "polite",
						class: "mt-3 flex flex-col gap-2 text-xs text-on-surface-variant"
					})]
				}),
				/* @__PURE__ */ u("section", {
					class: "soft-panel p-4 mt-3",
					children: [/* @__PURE__ */ u("h2", {
						class: "panel-title",
						children: "存储使用"
					}), /* @__PURE__ */ u("div", {
						id: "settings-storage-list",
						role: "status",
						"aria-live": "polite",
						class: "mt-3 flex flex-col gap-2 text-xs text-on-surface-variant",
						children: /* @__PURE__ */ u("div", {
							class: "rounded-lg bg-surface-container-low px-3 py-2",
							children: "正在读取本地存储…"
						})
					})]
				}),
				/* @__PURE__ */ u("section", {
					class: "soft-panel p-4 mt-3",
					children: [/* @__PURE__ */ u("h2", {
						class: "panel-title",
						children: "资料范围"
					}), /* @__PURE__ */ u("div", {
						class: "mt-3 flex flex-col gap-2 text-xs text-on-surface-variant",
						children: [
							/* @__PURE__ */ u("div", {
								class: "rounded-lg bg-surface-container-low px-3 py-2",
								children: [/* @__PURE__ */ u("div", {
									class: "font-semibold text-on-surface",
									children: "监控目录"
								}), /* @__PURE__ */ u("div", {
									class: "mt-1 text-[11px] leading-relaxed text-on-surface-variant/70",
									children: "系统会读取已添加的本地文件夹。"
								})]
							}),
							/* @__PURE__ */ u("div", {
								class: "rounded-lg bg-surface-container-low px-3 py-2",
								children: [/* @__PURE__ */ u("div", {
									class: "font-semibold text-on-surface",
									children: "采集内容"
								}), /* @__PURE__ */ u("div", {
									class: "mt-1 text-[11px] leading-relaxed text-on-surface-variant/70",
									children: "网页、临时笔记和知识产物会进入文件库。"
								})]
							}),
							/* @__PURE__ */ u("button", {
								onClick: call("switchView", "library"),
								class: "toolbar-btn justify-between",
								children: [/* @__PURE__ */ u("span", { children: "查看文件库" }), /* @__PURE__ */ u(Icon, { name: "arrow_forward" })]
							})
						]
					})]
				})
			]
		});
	}
	function SettingsView() {
		return /* @__PURE__ */ u("div", {
			id: settingsViewContract.viewId,
			class: "view hidden flex flex-col flex-1 min-h-0 overflow-hidden",
			role: "region",
			"aria-labelledby": "settings-title",
			tabIndex: -1,
			children: [/* @__PURE__ */ u(SettingsHeader, {}), /* @__PURE__ */ u("div", {
				class: "workspace-shell",
				children: /* @__PURE__ */ u("div", {
					class: "workspace-card workspace-grid",
					children: [/* @__PURE__ */ u("section", {
						class: "overflow-y-auto custom-scrollbar p-5",
						children: /* @__PURE__ */ u("div", {
							class: "grid grid-cols-1 xl:grid-cols-2 gap-5 items-start",
							children: [/* @__PURE__ */ u("div", {
								class: "flex flex-col gap-5",
								children: [/* @__PURE__ */ u(HealthPanel, {}), /* @__PURE__ */ u(SourcesPanel, {})]
							}), /* @__PURE__ */ u("div", {
								class: "flex flex-col gap-5",
								children: [/* @__PURE__ */ u(ModelPanel, {}), /* @__PURE__ */ u(PreferencePanel, {})]
							})]
						})
					}), /* @__PURE__ */ u(ContextPanel, {})]
				})
			})]
		});
	}
	function mountSettingsView() {
		const root = document.getElementById(settingsViewContract.rootId);
		if (!root) return;
		R(/* @__PURE__ */ u(SettingsView, {}), root);
		win.applyI18n?.();
		win.renderLocalIcons?.(root);
	}
	win.DocFlowSettingsApp = { mountSettingsView };
	mountSettingsView();
	//#endregion
	exports.mountSettingsView = mountSettingsView;
	return exports;
})({});

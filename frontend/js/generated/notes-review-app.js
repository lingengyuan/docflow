var DocFlowNotesReviewApp = (function(exports) {
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
	//#region node_modules/preact/jsx-runtime/dist/jsxRuntime.module.js
	var f = 0;
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
	//#region frontend/src/notes-review-app.tsx
	var win = typeof window === "undefined" ? void 0 : window;
	function asRecord(value) {
		return value && typeof value === "object" && !Array.isArray(value) ? value : {};
	}
	function asList(value) {
		return Array.isArray(value) ? value.filter((item) => Boolean(item) && typeof item === "object") : [];
	}
	function asReviewItems(value) {
		return asList(value).map((item) => ({
			file: asRecord(item.file),
			reason: stringValue(item.reason),
			priority: numberValue(item.priority),
			keywords: Array.isArray(item.keywords) ? item.keywords.map((word) => String(word)) : []
		}));
	}
	function numberValue(value) {
		const parsed = Number(value || 0);
		return Number.isFinite(parsed) ? parsed : 0;
	}
	function stringValue(value, fallback = "") {
		return typeof value === "string" && value ? value : fallback;
	}
	function firstFile(item) {
		return asList(item.files)[0] || {};
	}
	function fileId(file) {
		return numberValue(file.id);
	}
	function buildRelationshipOpportunityAction(item) {
		const data = asRecord(item);
		const source = asRecord(data.source);
		const target = asRecord(data.target);
		const sourceId = fileId(source);
		const targetId = fileId(target);
		const terms = Array.isArray(data.shared_terms) ? data.shared_terms.slice(0, 4).map(String) : [];
		return {
			source,
			target,
			sourceId,
			targetId,
			previewId: sourceId || targetId,
			canSave: sourceId > 0 && targetId > 0,
			terms
		};
	}
	function openFile(file) {
		const id = fileId(file);
		if (id > 0) win?.openFilePreview?.(id);
	}
	function Icon({ name, className = "", size = "14px" }) {
		return /* @__PURE__ */ u("span", {
			class: `material-symbols-outlined ${className}`,
			style: { fontSize: size },
			children: name
		});
	}
	function buildKnowledgeReviewViewModel(review) {
		const data = asRecord(review);
		const depth = asRecord(data.knowledge_depth);
		const queue = asReviewItems(data.review_queue);
		const workflow = asRecord(data.workflow);
		const steps = asList(workflow.steps).map((step) => ({
			id: stringValue(step.id, "step"),
			title: stringValue(step.title, "步骤"),
			count: numberValue(step.count),
			complete: Boolean(step.complete),
			detail: stringValue(step.detail),
			next_action: stringValue(step.next_action)
		}));
		return {
			countLabel: queue.length ? `${queue.length} 项` : "",
			empty: !review,
			signals: asRecord(data.signals),
			workflow: {
				steps,
				completed: numberValue(workflow.completed),
				total: numberValue(workflow.total || steps.length || 1),
				next_step: asRecord(workflow.next_step)
			},
			concepts: asList(depth.concepts),
			trails: asList(depth.source_trails),
			gaps: asList(depth.coverage_gaps),
			opportunities: asList(depth.relationship_opportunities),
			queue,
			recommendations: asList(data.recommendations),
			relationships: asList(data.relationship_timeline),
			depthActions: asList(depth.next_actions),
			topics: asList(data.topic_activity)
		};
	}
	function SignalCard({ label, value }) {
		return /* @__PURE__ */ u("div", {
			class: "rounded-lg bg-surface-container-low px-3 py-2",
			children: [/* @__PURE__ */ u("div", {
				class: "text-[11px] text-on-surface-variant/60",
				children: label
			}), /* @__PURE__ */ u("div", {
				class: "mt-0.5 text-sm font-semibold text-on-surface",
				children: numberValue(value)
			})]
		});
	}
	function WorkflowCard({ workflow }) {
		if (!workflow.steps.length) return null;
		const nextStep = asRecord(workflow.next_step);
		const next = stringValue(nextStep.detail, stringValue(nextStep.next_action, "继续导入资料并提问"));
		return /* @__PURE__ */ u("section", {
			class: "mt-3 rounded-xl border border-outline-variant/50 bg-surface-container-lowest px-3 py-3",
			"aria-label": "知识闭环",
			children: [
				/* @__PURE__ */ u("div", {
					class: "flex items-start justify-between gap-3",
					children: [/* @__PURE__ */ u("div", { children: [/* @__PURE__ */ u("div", {
						class: "text-[11px] font-bold text-on-surface",
						children: "知识闭环"
					}), /* @__PURE__ */ u("div", {
						class: "mt-0.5 text-[11px] text-on-surface-variant/65",
						children: "资料、问题、来源、笔记、关联和反馈放在同一条回顾线上。"
					})] }), /* @__PURE__ */ u("span", {
						class: "rounded-full bg-primary-container px-2 py-0.5 text-[11px] font-semibold text-primary",
						children: [
							workflow.completed,
							"/",
							workflow.total
						]
					})]
				}),
				/* @__PURE__ */ u("div", {
					class: "mt-3 grid grid-cols-2 gap-2",
					children: workflow.steps.map((step) => /* @__PURE__ */ u("div", {
						class: "rounded-lg bg-surface-container-low px-3 py-2",
						children: [/* @__PURE__ */ u("div", {
							class: "flex items-center gap-2",
							children: [
								/* @__PURE__ */ u(Icon, {
									name: step.complete ? "check_circle" : "radio_button_unchecked",
									className: `${step.complete ? "bg-primary text-on-primary" : "bg-surface-container text-on-surface-variant"} rounded-full`
								}),
								/* @__PURE__ */ u("span", {
									class: "text-[11px] font-semibold text-on-surface",
									children: step.title
								}),
								/* @__PURE__ */ u("span", {
									class: "ml-auto text-[11px] text-on-surface-variant/60",
									children: step.count
								})
							]
						}), /* @__PURE__ */ u("div", {
							class: "mt-1 text-[10px] leading-relaxed text-on-surface-variant/65",
							children: step.detail
						})]
					}, step.id))
				}),
				/* @__PURE__ */ u("div", {
					class: "mt-3 rounded-lg bg-secondary-container/80 px-3 py-2 text-[11px] text-on-surface",
					children: [/* @__PURE__ */ u("span", {
						class: "font-semibold",
						children: "下一步："
					}), next]
				})
			]
		});
	}
	function TopicPills({ topics }) {
		if (!topics.length) return null;
		return /* @__PURE__ */ u("div", {
			class: "mt-3 flex flex-wrap gap-2",
			children: topics.slice(0, 3).map((topic) => /* @__PURE__ */ u("span", {
				class: "inline-flex items-center rounded-full bg-surface-container-low px-2 py-1 text-[11px] font-semibold text-on-surface-variant",
				children: [
					stringValue(topic.title, "主题"),
					" · ",
					numberValue(topic.file_count)
				]
			}))
		});
	}
	function ConceptList({ concepts }) {
		if (!concepts.length) return null;
		return /* @__PURE__ */ u("div", {
			class: "mt-3",
			children: [/* @__PURE__ */ u("div", {
				class: "mb-2 text-[11px] font-bold text-on-surface-variant/60",
				children: "活跃概念"
			}), /* @__PURE__ */ u("div", {
				class: "flex flex-wrap gap-2",
				children: concepts.slice(0, 4).map((item) => /* @__PURE__ */ u("button", {
					onClick: () => openFile(firstFile(item)),
					class: "rounded-full bg-surface-container-low px-3 py-1.5 text-left text-[11px] font-semibold text-on-surface-variant hover:bg-surface-container transition-colors",
					children: [
						stringValue(item.title, "概念"),
						" · ",
						numberValue(item.file_count),
						" 份资料 · ",
						numberValue(item.question_count),
						" 次提问"
					]
				}))
			})]
		});
	}
	function ReviewQueue({ queue }) {
		if (!queue.length) return /* @__PURE__ */ u("div", {
			class: "rounded-lg bg-surface-container-low px-3 py-3",
			children: "导入资料并保存回答后，这里会出现回顾建议。"
		});
		return /* @__PURE__ */ u(S, { children: queue.slice(0, 3).map((item) => {
			const file = asRecord(item.file);
			return /* @__PURE__ */ u("button", {
				onClick: () => openFile(file),
				class: "w-full text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors",
				children: [/* @__PURE__ */ u("div", {
					class: "flex items-start justify-between gap-3",
					children: [/* @__PURE__ */ u("div", {
						class: "min-w-0",
						children: [/* @__PURE__ */ u("div", {
							class: "font-semibold text-on-surface line-clamp-1",
							children: stringValue(file.file_name, "资料")
						}), /* @__PURE__ */ u("div", {
							class: "mt-1 text-[11px] text-on-surface-variant/60",
							children: item.reason || "值得回顾"
						})]
					}), /* @__PURE__ */ u("span", {
						class: "rounded-full bg-primary-container px-2 py-0.5 text-[11px] font-semibold text-primary",
						children: numberValue(item.priority)
					})]
				}), item.keywords?.length ? /* @__PURE__ */ u("div", {
					class: "mt-2 flex flex-wrap gap-1",
					children: item.keywords.slice(0, 3).map((word) => /* @__PURE__ */ u("span", {
						class: "rounded-full bg-surface-container px-2 py-0.5 text-[10px] text-on-surface-variant",
						children: word
					}))
				}) : null]
			});
		}) });
	}
	function RelationshipList({ relationships }) {
		if (!relationships.length) return null;
		return /* @__PURE__ */ u("div", {
			class: "mt-3 flex flex-col gap-2",
			children: relationships.slice(0, 3).map((item) => {
				const note = asRecord(item.note);
				const source = asRecord(item.source);
				return /* @__PURE__ */ u("button", {
					onClick: () => openFile(fileId(source) ? source : note),
					class: "w-full text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors",
					children: [
						/* @__PURE__ */ u("div", {
							class: "flex items-center gap-2 text-[11px] font-semibold text-primary",
							children: [/* @__PURE__ */ u(Icon, { name: "account_tree" }), stringValue(item.label, "知识关联")]
						}),
						/* @__PURE__ */ u("div", {
							class: "mt-1 text-xs text-on-surface line-clamp-1",
							children: stringValue(note.file_name, "保存内容")
						}),
						/* @__PURE__ */ u("div", {
							class: "mt-0.5 text-[11px] text-on-surface-variant/65 line-clamp-1",
							children: ["来源：", stringValue(source.file_name, "资料")]
						})
					]
				});
			})
		});
	}
	function SourceTrailList({ trails }) {
		if (!trails.length) return null;
		return /* @__PURE__ */ u("div", {
			class: "mt-3",
			children: [/* @__PURE__ */ u("div", {
				class: "mb-2 text-[11px] font-bold text-on-surface-variant/60",
				children: "来源轨迹"
			}), /* @__PURE__ */ u("div", {
				class: "flex flex-col gap-2",
				children: trails.slice(0, 3).map((item) => {
					const file = firstFile(item);
					const feedback = asRecord(item.feedback);
					const rating = feedback.rating === "useful" ? "已标记有用" : feedback.rating === "not_useful" ? "需要改进" : "未反馈";
					return /* @__PURE__ */ u("button", {
						onClick: () => openFile(file),
						class: "w-full text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors",
						children: [
							/* @__PURE__ */ u("div", {
								class: "flex items-center gap-2 text-[11px] font-semibold text-primary",
								children: [/* @__PURE__ */ u(Icon, { name: "route" }), "问题引用了来源"]
							}),
							/* @__PURE__ */ u("div", {
								class: "mt-1 text-xs text-on-surface line-clamp-1",
								children: stringValue(item.question, "最近问题")
							}),
							/* @__PURE__ */ u("div", {
								class: "mt-0.5 text-[11px] text-on-surface-variant/65 line-clamp-1",
								children: [
									stringValue(file.file_name, "来源资料"),
									" · ",
									rating
								]
							})
						]
					});
				})
			})]
		});
	}
	function CoverageGapList({ gaps }) {
		if (!gaps.length) return null;
		return /* @__PURE__ */ u("div", {
			class: "mt-3",
			children: [/* @__PURE__ */ u("div", {
				class: "mb-2 text-[11px] font-bold text-on-surface-variant/60",
				children: "待补齐"
			}), /* @__PURE__ */ u("div", {
				class: "flex flex-col gap-2",
				children: gaps.slice(0, 3).map((item) => {
					const file = asRecord(item.file);
					return /* @__PURE__ */ u("button", {
						onClick: () => openFile(file),
						class: "w-full text-left rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors",
						children: /* @__PURE__ */ u("div", {
							class: "flex items-start justify-between gap-3",
							children: [/* @__PURE__ */ u("div", {
								class: "min-w-0",
								children: [
									/* @__PURE__ */ u("div", {
										class: "font-semibold text-on-surface line-clamp-1",
										children: stringValue(item.title, "补齐资料")
									}),
									/* @__PURE__ */ u("div", {
										class: "mt-1 text-[11px] text-on-surface-variant/65",
										children: stringValue(item.detail)
									}),
									/* @__PURE__ */ u("div", {
										class: "mt-1 text-[11px] text-on-surface-variant/55 line-clamp-1",
										children: stringValue(file.file_name, "资料")
									})
								]
							}), /* @__PURE__ */ u("span", {
								class: "rounded-full bg-tertiary-container px-2 py-0.5 text-[11px] font-semibold text-tertiary",
								children: numberValue(item.priority)
							})]
						})
					});
				})
			})]
		});
	}
	function RelationshipOpportunityList({ opportunities }) {
		if (!opportunities.length) return null;
		return /* @__PURE__ */ u("div", {
			class: "mt-3",
			children: [/* @__PURE__ */ u("div", {
				class: "mb-2 text-[11px] font-bold text-on-surface-variant/60",
				children: "可连接资料"
			}), /* @__PURE__ */ u("div", {
				class: "flex flex-col gap-2",
				children: opportunities.slice(0, 3).map((item) => {
					const action = buildRelationshipOpportunityAction(item);
					return /* @__PURE__ */ u("div", {
						class: "w-full rounded-lg bg-surface-container-low px-3 py-3",
						children: [
							/* @__PURE__ */ u("div", {
								class: "flex items-center gap-2 text-[11px] font-semibold text-primary",
								children: [/* @__PURE__ */ u(Icon, { name: "hub" }), "建议建立资料关联"]
							}),
							/* @__PURE__ */ u("div", {
								class: "mt-1 text-xs text-on-surface line-clamp-1",
								children: [
									stringValue(action.source.file_name, "资料"),
									" ↔ ",
									stringValue(action.target.file_name, "资料")
								]
							}),
							/* @__PURE__ */ u("div", {
								class: "mt-0.5 text-[11px] text-on-surface-variant/65 line-clamp-1",
								children: ["共同线索：", action.terms.join(" · ") || "内容相近"]
							}),
							/* @__PURE__ */ u("div", {
								class: "mt-2 flex items-center gap-2",
								children: [/* @__PURE__ */ u("button", {
									onClick: () => openFile({ id: action.previewId }),
									class: "toolbar-btn !h-8",
									title: "查看资料",
									"aria-label": "查看资料",
									children: [/* @__PURE__ */ u(Icon, { name: "article" }), "查看"]
								}), /* @__PURE__ */ u("button", {
									onClick: (event) => confirmKnowledgeRelationship(action.sourceId, action.targetId, event.currentTarget),
									disabled: !action.canSave,
									class: "toolbar-btn toolbar-btn-primary !h-8",
									title: "保存关联",
									"aria-label": "保存关联",
									children: [/* @__PURE__ */ u(Icon, { name: "add_link" }), "保存关联"]
								})]
							})
						]
					});
				})
			})]
		});
	}
	function ActionList({ items, tone }) {
		if (!items.length) return null;
		const className = tone === "primary" ? "w-full text-left rounded-lg bg-primary-container/70 px-3 py-2 text-xs text-on-surface hover:bg-primary-container transition-colors" : "w-full text-left rounded-lg bg-secondary-container/80 px-3 py-2 text-xs text-on-surface hover:bg-secondary-container transition-colors";
		return /* @__PURE__ */ u("div", {
			class: "mt-3 flex flex-col gap-2",
			children: items.slice(0, tone === "primary" ? 2 : 2).map((item) => /* @__PURE__ */ u("button", {
				onClick: () => openFile({ id: item.file_id }),
				class: className,
				children: [/* @__PURE__ */ u("div", {
					class: "font-semibold",
					children: stringValue(item.title, "下一步")
				}), /* @__PURE__ */ u("div", {
					class: "mt-0.5 text-[11px] text-on-surface-variant",
					children: stringValue(item.detail)
				})]
			}))
		});
	}
	function NotesReviewPanel({ review }) {
		const model = buildKnowledgeReviewViewModel(review);
		if (model.empty) return /* @__PURE__ */ u("div", {
			class: "rounded-lg bg-surface-container-low px-3 py-3",
			children: "回顾建议暂时不可用。"
		});
		return /* @__PURE__ */ u(S, { children: [
			/* @__PURE__ */ u("div", {
				class: "grid grid-cols-3 gap-2",
				children: [
					/* @__PURE__ */ u(SignalCard, {
						label: "资料",
						value: model.signals.files
					}),
					/* @__PURE__ */ u(SignalCard, {
						label: "问题",
						value: model.signals.questions
					}),
					/* @__PURE__ */ u(SignalCard, {
						label: "关联",
						value: numberValue(model.signals.backlinks) + numberValue(model.signals.source_links)
					})
				]
			}),
			/* @__PURE__ */ u(WorkflowCard, { workflow: model.workflow }),
			/* @__PURE__ */ u(ConceptList, { concepts: model.concepts }),
			/* @__PURE__ */ u(CoverageGapList, { gaps: model.gaps }),
			/* @__PURE__ */ u(SourceTrailList, { trails: model.trails }),
			/* @__PURE__ */ u(RelationshipOpportunityList, { opportunities: model.opportunities }),
			/* @__PURE__ */ u("div", {
				class: "mt-3 flex flex-col gap-2",
				children: /* @__PURE__ */ u(ReviewQueue, { queue: model.queue })
			}),
			/* @__PURE__ */ u(RelationshipList, { relationships: model.relationships }),
			/* @__PURE__ */ u(ActionList, {
				items: model.depthActions,
				tone: "secondary"
			}),
			/* @__PURE__ */ u(ActionList, {
				items: model.recommendations,
				tone: "primary"
			}),
			/* @__PURE__ */ u(TopicPills, { topics: model.topics })
		] });
	}
	async function confirmKnowledgeRelationship(sourceId, targetId, button) {
		if (!win || !sourceId || !targetId || !button) return;
		const previous = button.innerHTML;
		button.disabled = true;
		button.innerHTML = "<span class=\"spinner\"></span><span class=\"ml-1.5\">保存中…</span>";
		try {
			const response = await fetch(`${win.API || ""}/api/knowledge/relationships`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					source_file_id: sourceId,
					target_file_id: targetId,
					relation: "manual_relationship"
				})
			});
			if (!response.ok) {
				const message = await win.responseUserMessage?.(response, "资料关联保存失败，请稍后再试。");
				throw new Error(message || "资料关联保存失败，请稍后再试。");
			}
			button.innerHTML = "<span class=\"material-symbols-outlined\" style=\"font-size:14px\">done</span><span class=\"ml-1\">已保存</span>";
			await Promise.all([win.refreshNotesView?.() || Promise.resolve(), win.loadKnowledgeOverview?.(win.activeLibraryFileId || null) || Promise.resolve()]);
		} catch (error) {
			button.disabled = false;
			button.innerHTML = previous;
			const panel = document.getElementById("knowledge-review-panel");
			const raw = error instanceof Error ? error.message : String(error);
			const message = win.userFacingErrorMessage?.(raw, "资料关联保存失败，请稍后再试。") || "资料关联保存失败，请稍后再试。";
			const safe = win.escHtml?.(message) || message.replace(/[&<>"']/g, (ch) => ({
				"&": "&amp;",
				"<": "&lt;",
				">": "&gt;",
				"\"": "&quot;",
				"'": "&#39;"
			})[ch] || ch);
			panel?.insertAdjacentHTML("afterbegin", `<div class="mb-2 rounded-lg bg-error/10 px-3 py-2 text-[11px] font-bold text-error">保存失败：${safe}</div>`);
		}
	}
	function renderKnowledgeReview(review) {
		if (!win) return;
		const panel = document.getElementById("knowledge-review-panel");
		const count = document.getElementById("knowledge-review-count");
		if (!panel) return;
		const model = buildKnowledgeReviewViewModel(review);
		if (count) count.textContent = model.countLabel;
		R(/* @__PURE__ */ u(NotesReviewPanel, { review }), panel);
		win.renderLocalIcons?.(panel);
	}
	if (win) win.DocFlowNotesReviewApp = {
		renderKnowledgeReview,
		confirmKnowledgeRelationship
	};
	//#endregion
	exports.buildKnowledgeReviewViewModel = buildKnowledgeReviewViewModel;
	exports.buildRelationshipOpportunityAction = buildRelationshipOpportunityAction;
	exports.confirmKnowledgeRelationship = confirmKnowledgeRelationship;
	exports.renderKnowledgeReview = renderKnowledgeReview;
	return exports;
})({});

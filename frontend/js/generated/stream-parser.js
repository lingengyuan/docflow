var DocFlowStreamParser = (function(exports) {
	Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
	//#region frontend/src/stream-parser.ts
	function consumeSseBuffer(input) {
		const parts = input.split("\n\n");
		const buffer = parts.pop() ?? "";
		const events = [];
		for (const part of parts) {
			let event = "";
			const dataLines = [];
			for (const line of part.split("\n")) if (line.startsWith("event: ")) event = line.slice(7);
			else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
			if (event) events.push({
				event,
				data: dataLines.join("\n")
			});
		}
		return {
			events,
			buffer
		};
	}
	//#endregion
	exports.consumeSseBuffer = consumeSseBuffer;
	return exports;
})({});

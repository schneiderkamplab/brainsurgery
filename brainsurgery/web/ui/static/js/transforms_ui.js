function createTransformsUI({
  appState,
  transformsEl,
  transformSearchEl,
  loadPanel,
  optionsEmpty,
  transformPanel,
  transformTitle,
  transformFields,
  transformRunBtn,
  setStatus,
  stopProgress,
  buildReferenceFromModel,
  copyFromFilterToToTemplate,
}) {
  let onSelectionChanged = () => {};

  function setOnSelectionChanged(callback) {
    onSelectionChanged = callback;
  }

  function getTransformMeta(name) {
    return appState.allTransforms.find((t) => t.name === name) || null;
  }

  function isReadyTransform(name) {
    const meta = getTransformMeta(name);
    return !!(meta && meta.enabled);
  }

  function isRunnableTransform(name) {
    return !!name && name !== "load" && isReadyTransform(name);
  }

  function getIsIteratingTransform(name) {
    const meta = getTransformMeta(name);
    return !!(meta && meta.iterating);
  }

  function getTransformConfig(name) {
    if (!appState.transformConfigByName[name]) {
      appState.transformConfigByName[name] = { fields: {}, save_mode: "server", save_download_format: "safetensors" };
    }
    return appState.transformConfigByName[name];
  }

  function resetTransformSearch() {
    transformSearchEl.value = "";
    renderTransforms();
  }

  function commitRefFromModel(key, alias, filterText) {
    const meta = getTransformMeta(appState.selectedTransform);
    if (!meta) return;
    const cfg = getTransformConfig(appState.selectedTransform);
    cfg.fields[key] = buildReferenceFromModel(alias, filterText);
    if (meta.kind === "binary" && key === "from") {
      const templ = copyFromFilterToToTemplate((filterText || "").trim());
      cfg.fields.to = alias + "::" + (templ || ".*");
    }
    if (meta.kind === "ternary" && key === "from_a") {
      const templ = copyFromFilterToToTemplate((filterText || "").trim());
      const expr = templ || ".*";
      cfg.fields.from_b = alias + "::" + expr;
      cfg.fields.to = alias + "::" + expr;
    }
    renderTransformPanel();
    setStatus("Committed " + key + " for " + appState.selectedTransform + " from " + alias + ".");
  }

  function renderTransformPanel() {
    if (!isRunnableTransform(appState.selectedTransform)) {
      transformPanel.classList.add("hidden");
      return;
    }
    const meta = getTransformMeta(appState.selectedTransform);
    const cfg = getTransformConfig(appState.selectedTransform);
    const modeKey = typeof meta.mode_key === "string" && meta.mode_key ? meta.mode_key : null;
    const modeNames = Array.isArray(meta.modes) ? meta.modes.map((m) => String(m)) : [];
    const defaultMode =
      typeof meta.default_mode === "string" && meta.default_mode
        ? String(meta.default_mode)
        : (modeNames[0] || "default");
    const rawMode = modeKey ? String(cfg.fields[modeKey] == null ? defaultMode : cfg.fields[modeKey]).trim().toLowerCase() : "";
    const mode = modeKey
      ? (modeNames.map((m) => m.toLowerCase()).includes(rawMode) ? rawMode : String(defaultMode).toLowerCase())
      : null;
    if (modeKey) cfg.fields[modeKey] = mode;
    const allowedByMode = meta.mode_allowed_keys && typeof meta.mode_allowed_keys === "object"
      ? meta.mode_allowed_keys
      : {};
    const requiredByMode = meta.mode_required_keys && typeof meta.mode_required_keys === "object"
      ? meta.mode_required_keys
      : {};
    const allowed = modeKey
      ? (Array.isArray(allowedByMode[mode]) ? allowedByMode[mode].map((k) => String(k)) : [])
      : (Array.isArray(meta.allowed_keys) ? meta.allowed_keys : []);
    const required = new Set(
      modeKey
        ? (Array.isArray(requiredByMode[mode]) ? requiredByMode[mode].map((k) => String(k)) : [])
        : (Array.isArray(meta.required_keys) ? meta.required_keys : [])
    );
    const refKeys = Array.isArray(meta.reference_keys) ? meta.reference_keys : [];
    const refSet = new Set(refKeys);
    const booleanKeys = new Set(Array.isArray(meta.boolean_keys) ? meta.boolean_keys : []);
    const orderedKeys = [
      ...refKeys.filter((k) => k !== modeKey),
      ...allowed.filter((k) => !refSet.has(k) && k !== modeKey),
    ];
    transformTitle.textContent = appState.selectedTransform;
    transformFields.innerHTML = "";
    transformRunBtn.textContent = "Run " + appState.selectedTransform;

    if (modeKey) {
      const modeSelect = document.createElement("select");
      const options = modeNames.length ? modeNames : [String(defaultMode)];
      modeSelect.innerHTML = options
        .map((name) => "<option value='" + String(name).toLowerCase() + "'>" + modeKey + ": " + String(name).toLowerCase() + "</option>")
        .join("");
      modeSelect.value = String(mode || defaultMode).toLowerCase();
      modeSelect.addEventListener("change", () => {
        cfg.fields[modeKey] = modeSelect.value;
        const nextAllowed = new Set(
          Array.isArray(allowedByMode[modeSelect.value]) ? allowedByMode[modeSelect.value].map((k) => String(k)) : []
        );
        for (const key of Object.keys(cfg.fields)) {
          if (key === modeKey) continue;
          if (!nextAllowed.has(key)) {
            delete cfg.fields[key];
          }
        }
        renderTransformPanel();
      });
      transformFields.appendChild(modeSelect);
    }

    if (appState.selectedTransform === "set") {
      const current = document.createElement("div");
      current.className = "binary-summary";
      const line = document.createElement("div");
      line.className = "value";
      line.style.marginBottom = "0";
      line.textContent =
        "dry-run=" + String(Boolean(appState.latestRuntimeFlags.dry_run)) +
        ", preview=" + String(Boolean(appState.latestRuntimeFlags.preview)) +
        ", verbose=" + String(Boolean(appState.latestRuntimeFlags.verbose));
      current.appendChild(line);
      transformFields.appendChild(current);
    }

    if (appState.selectedTransform !== "assert") {
      for (const key of orderedKeys) {
      if (appState.selectedTransform === "dump" && key === "format") {
        const fmtSelect = document.createElement("select");
        fmtSelect.innerHTML =
          "<option value='compact'>format: compact</option>" +
          "<option value='tree'>format: tree</option>" +
          "<option value='json'>format: json</option>";
        fmtSelect.value = cfg.fields.format == null ? "compact" : String(cfg.fields.format).toLowerCase();
        if (!["compact", "tree", "json"].includes(fmtSelect.value)) fmtSelect.value = "compact";
        fmtSelect.addEventListener("change", () => { cfg.fields.format = fmtSelect.value; });
        transformFields.appendChild(fmtSelect);
        continue;
      }
      if (appState.selectedTransform === "dump" && key === "verbosity") {
        const verbositySelect = document.createElement("select");
        verbositySelect.innerHTML =
          "<option value='shape'>verbosity: shape</option>" +
          "<option value='stat'>verbosity: stat</option>" +
          "<option value='full'>verbosity: full</option>";
        verbositySelect.value = cfg.fields.verbosity == null ? "shape" : String(cfg.fields.verbosity).toLowerCase();
        if (!["shape", "stat", "full"].includes(verbositySelect.value)) verbositySelect.value = "shape";
        verbositySelect.addEventListener("change", () => { cfg.fields.verbosity = verbositySelect.value; });
        transformFields.appendChild(verbositySelect);
        continue;
      }
      if (appState.selectedTransform === "execute" && key === "plan-yaml") {
        const label = document.createElement("div");
        label.className = "binary-summary";
        label.innerHTML =
          "<div class='label'>plan-yaml</div>" +
          "<div class='value'>Paste a YAML plan or transforms list.</div>";
        transformFields.appendChild(label);

        const textarea = document.createElement("textarea");
        textarea.placeholder =
          "transforms:\n" +
          "  - dump: { target: model::.*, format: compact }\n" +
          "  - exit: {}";
        textarea.rows = 8;
        textarea.value = cfg.fields[key] == null ? "" : String(cfg.fields[key]);
        textarea.addEventListener("input", () => { cfg.fields[key] = textarea.value; });
        transformFields.appendChild(textarea);
        continue;
      }
      if (booleanKeys.has(key)) {
        const boolLabel = document.createElement("label");
        boolLabel.className = "checkbox-row";
        const boolInput = document.createElement("input");
        boolInput.type = "checkbox";
        const current = cfg.fields[key];
        if (typeof current === "boolean") {
          boolInput.checked = current;
          boolInput.indeterminate = false;
        } else {
          boolInput.checked = false;
          boolInput.indeterminate = true;
        }
        boolInput.addEventListener("change", () => {
          boolInput.indeterminate = false;
          cfg.fields[key] = boolInput.checked;
        });
        const boolText = document.createElement("span");
        const suffix = required.has(key) ? "required" : "optional";
        boolText.textContent = key + " (" + suffix + ")";
        boolLabel.appendChild(boolInput);
        boolLabel.appendChild(boolText);
        transformFields.appendChild(boolLabel);
        continue;
      }
      const input = document.createElement("input");
      const suffix = required.has(key) ? "required" : "optional";
      input.placeholder = key + " (" + suffix + ")";
      input.value = cfg.fields[key] == null ? "" : String(cfg.fields[key]);
      input.addEventListener("input", () => { cfg.fields[key] = input.value; });
        transformFields.appendChild(input);
      }
    }

    if (appState.selectedTransform === "help") {
      const commandSelect = document.createElement("select");
      const helpCommands = Array.isArray(meta.help_commands) ? meta.help_commands : [];
      commandSelect.innerHTML =
        "<option value=''>command: all commands</option>" +
        helpCommands.map((name) => "<option value='" + name + "'>command: " + name + "</option>").join("");
      commandSelect.value = cfg.fields.help_command == null ? "" : String(cfg.fields.help_command);
      if (!["", ...helpCommands].includes(commandSelect.value)) commandSelect.value = "";
      commandSelect.addEventListener("change", () => {
        cfg.fields.help_command = commandSelect.value;
        if (commandSelect.value !== "assert") cfg.fields.help_subcommand = "";
        renderTransformPanel();
      });
      transformFields.appendChild(commandSelect);

      const subcommandSelect = document.createElement("select");
      const subcommandsByCommand =
        (meta.help_subcommands && typeof meta.help_subcommands === "object")
          ? meta.help_subcommands
          : {};
      const selectedCommand = String(cfg.fields.help_command || "");
      const commandSubcommands = selectedCommand
        ? (Array.isArray(subcommandsByCommand[selectedCommand]) ? subcommandsByCommand[selectedCommand] : [])
        : [];
      subcommandSelect.innerHTML =
        "<option value=''>subcommand: none</option>" +
        commandSubcommands.map((name) => "<option value='" + name + "'>subcommand: " + name + "</option>").join("");
      subcommandSelect.value = cfg.fields.help_subcommand == null ? "" : String(cfg.fields.help_subcommand);
      if (!["", ...commandSubcommands].includes(subcommandSelect.value)) subcommandSelect.value = "";
      subcommandSelect.disabled = !selectedCommand || commandSubcommands.length === 0;
      subcommandSelect.addEventListener("change", () => { cfg.fields.help_subcommand = subcommandSelect.value; });
      transformFields.appendChild(subcommandSelect);
    }

    if (appState.selectedTransform === "assert") {
      const yamlInput = document.createElement("textarea");
      yamlInput.rows = 9;
      yamlInput.placeholder =
        "YAML assert expression\n\n" +
        "equal:\n" +
        "  left: model::a.weight\n" +
        "  right: model::b.weight\n\n" +
        "Nested example:\n" +
        "all:\n" +
        "  - exists: model::.*weight\n" +
        "  - not:\n" +
        "      equal:\n" +
        "        left: model::a.weight\n" +
        "        right: model::b.weight";
      yamlInput.value = cfg.fields.assert_yaml == null ? "" : String(cfg.fields.assert_yaml);
      yamlInput.addEventListener("input", () => { cfg.fields.assert_yaml = yamlInput.value; });
      transformFields.appendChild(yamlInput);
    }

    if (appState.selectedTransform === "exit") {
      const copyLabel = document.createElement("label");
      copyLabel.style.display = "flex";
      copyLabel.style.alignItems = "center";
      copyLabel.style.gap = "8px";
      copyLabel.style.marginBottom = "8px";
      const copyCheckbox = document.createElement("input");
      copyCheckbox.type = "checkbox";
      copyCheckbox.style.width = "auto";
      copyCheckbox.style.margin = "0";
      copyCheckbox.checked = Boolean(cfg.fields.exit_auto_copy);
      copyCheckbox.addEventListener("change", () => { cfg.fields.exit_auto_copy = copyCheckbox.checked; });
      const copyText = document.createElement("span");
      copyText.textContent = "Copy plan to clipboard";
      copyLabel.appendChild(copyCheckbox);
      copyLabel.appendChild(copyText);
      transformFields.appendChild(copyLabel);

      const summaryModeSelect = document.createElement("select");
      summaryModeSelect.innerHTML =
        "<option value='raw'>summary mode: raw</option>" +
        "<option value='resolve'>summary mode: resolve</option>";
      summaryModeSelect.value = cfg.fields.exit_summary_mode == null ? "raw" : String(cfg.fields.exit_summary_mode).toLowerCase();
      if (!["raw", "resolve"].includes(summaryModeSelect.value)) summaryModeSelect.value = "raw";
      summaryModeSelect.addEventListener("change", () => { cfg.fields.exit_summary_mode = summaryModeSelect.value; });
      transformFields.appendChild(summaryModeSelect);
    }

    if (appState.selectedTransform === "save") {
      const modeSelect = document.createElement("select");
      modeSelect.innerHTML = "<option value='server'>save on server path</option><option value='download'>download to browser</option>";
      modeSelect.value = cfg.save_mode || "server";
      modeSelect.addEventListener("change", () => {
        cfg.save_mode = modeSelect.value;
        renderTransformPanel();
      });
      transformFields.appendChild(modeSelect);

      if ((cfg.save_mode || "server") === "download") {
        const fmtSelect = document.createElement("select");
        fmtSelect.innerHTML = "<option value='safetensors'>download format: safetensors</option><option value='numpy'>download format: numpy</option><option value='torch'>download format: pytorch</option>";
        fmtSelect.value = cfg.save_download_format || "safetensors";
        fmtSelect.addEventListener("change", () => {
          cfg.save_download_format = fmtSelect.value;
        });
        transformFields.appendChild(fmtSelect);
      }
    }

    if (meta.kind === "binary" && refSet.has("from") && refSet.has("to")) {
      const copyBtn = document.createElement("button");
      copyBtn.className = "secondary-btn";
      copyBtn.textContent = "Copy from filter to to";
      copyBtn.addEventListener("click", () => {
        const fromRaw = String(cfg.fields.from || "");
        const sep = fromRaw.indexOf("::");
        if (sep < 0) {
          setStatus("Set from as alias::regex first.");
          return;
        }
        const alias = fromRaw.slice(0, sep);
        const expr = fromRaw.slice(sep + 2);
        const templ = copyFromFilterToToTemplate(expr);
        cfg.fields.to = alias + "::" + (templ || ".*");
        renderTransformPanel();
        setStatus("Copied from filter into to for " + appState.selectedTransform + ".");
      });
      transformFields.appendChild(copyBtn);
    }

    if (
      meta.kind === "ternary"
      && refSet.has("from_a")
      && refSet.has("from_b")
      && refSet.has("to")
    ) {
      const copyAButton = document.createElement("button");
      copyAButton.className = "secondary-btn";
      copyAButton.textContent = "Copy from_a filter to from_b + to";
      copyAButton.addEventListener("click", () => {
        const fromARaw = String(cfg.fields.from_a || "");
        const sep = fromARaw.indexOf("::");
        if (sep < 0) {
          setStatus("Set from_a as alias::regex first.");
          return;
        }
        const alias = fromARaw.slice(0, sep);
        const expr = fromARaw.slice(sep + 2);
        const templ = copyFromFilterToToTemplate(expr);
        const rewritten = templ || ".*";
        cfg.fields.from_b = alias + "::" + rewritten;
        cfg.fields.to = alias + "::" + rewritten;
        renderTransformPanel();
        setStatus("Copied from_a filter into from_b and to for " + appState.selectedTransform + ".");
      });
      transformFields.appendChild(copyAButton);

      const copyBButton = document.createElement("button");
      copyBButton.className = "secondary-btn";
      copyBButton.textContent = "Copy from_b filter to to";
      copyBButton.addEventListener("click", () => {
        const fromBRaw = String(cfg.fields.from_b || "");
        const sep = fromBRaw.indexOf("::");
        if (sep < 0) {
          setStatus("Set from_b as alias::regex first.");
          return;
        }
        const alias = fromBRaw.slice(0, sep);
        const expr = fromBRaw.slice(sep + 2);
        const templ = copyFromFilterToToTemplate(expr);
        cfg.fields.to = alias + "::" + (templ || ".*");
        renderTransformPanel();
        setStatus("Copied from_b filter into to for " + appState.selectedTransform + ".");
      });
      transformFields.appendChild(copyBButton);
    }

    transformPanel.classList.remove("hidden");
  }

  function updatePanels() {
    const showLoad = appState.selectedTransform === "load";
    const showTransform = isRunnableTransform(appState.selectedTransform);
    const hasSelection = !!appState.selectedTransform;
    loadPanel.classList.toggle("hidden", !showLoad);
    transformPanel.classList.toggle("hidden", !showTransform);
    optionsEmpty.classList.toggle("hidden", hasSelection);
    renderTransformPanel();
    if (appState.selectedTransform === "load") {
      stopProgress();
      setStatus("Load is selected. Pick a file to import a model.");
    } else if (!isReadyTransform(appState.selectedTransform)) {
      stopProgress();
      setStatus("Selected " + appState.selectedTransform + " is planned and not interactive yet.");
    } else if (!hasSelection) {
      stopProgress();
      setStatus("Ready.");
    } else {
      setStatus("Selected " + appState.selectedTransform + ".");
    }
  }

  function renderTransforms() {
    const query = transformSearchEl.value.trim().toLowerCase();
    const items = appState.allTransforms.filter((item) => item.name.toLowerCase().includes(query));
    transformsEl.innerHTML = "";
    if (!items.length) {
      const row = document.createElement("div");
      row.className = "transform-item";
      row.textContent = "No transforms match your search.";
      transformsEl.appendChild(row);
      return;
    }
    for (const item of items) {
      const row = document.createElement("div");
      row.className = "transform-item" + (item.enabled ? "" : " planned") + (appState.selectedTransform === item.name ? " selected" : "");
      row.dataset.transform = item.name;
      row.tabIndex = -1;
      const name = document.createElement("span");
      name.textContent = item.name;
      const badge = document.createElement("span");
      badge.className = "pill" + (item.enabled ? " enabled" : "");
      badge.textContent = item.enabled ? "ready" : "planned";
      row.appendChild(name);
      row.appendChild(badge);
      if (item.enabled) {
        row.addEventListener("click", () => {
          appState.selectedTransform = item.name;
          renderTransforms();
          updatePanels();
          onSelectionChanged();
        });
        row.addEventListener("keydown", (event) => {
          if (event.key !== "Enter" && event.key !== " ") return;
          event.preventDefault();
          row.click();
        });
      }
      transformsEl.appendChild(row);
    }
  }

  return {
    commitRefFromModel,
    getIsIteratingTransform,
    getTransformConfig,
    getTransformMeta,
    isRunnableTransform,
    renderTransforms,
    renderTransformPanel,
    resetTransformSearch,
    setOnSelectionChanged,
    updatePanels,
  };
}

export { createTransformsUI };

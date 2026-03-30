function createActions({
  appState,
  setStatus,
  progressController,
  getIsIteratingTransform,
  getTransformMeta,
  getTransformConfig,
  isRunnableTransform,
  resetTransformSearch,
  renderTransforms,
  updatePanels,
  renderModels,
  renderInsights,
  appendResultBlock,
  copyTextToClipboard,
  parseFieldValue,
  showAssertFailure,
  showPreviewConfirm,
  loadBtn,
  aliasInput,
  serverPathInput,
  fileInput,
  transformRunBtn,
}) {
  function _applyStateFromResponse(data) {
    appState.latestModels = data.models || [];
    if (data.runtime_flags && typeof data.runtime_flags === "object") {
      appState.latestRuntimeFlags = {
        dry_run: Boolean(data.runtime_flags.dry_run),
        preview: Boolean(data.runtime_flags.preview),
        verbose: Boolean(data.runtime_flags.verbose),
      };
    }
    if (data.insights && typeof data.insights === "object") {
      appState.latestInsights = data.insights;
    }
  }

  function _renderState() {
    renderModels(appState.latestModels);
    renderInsights();
    resetTransformSearch();
    renderTransforms();
    updatePanels();
  }

  function _downloadFromResponse(data) {
    const raw = atob(data.download_b64 || "");
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
    const blob = new Blob([bytes], { type: data.download_mime || "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = data.download_filename || "brainsurgery-save.bin";
    a.click();
    URL.revokeObjectURL(url);
  }

  async function _handleApplySuccess({ runTransformName, cfg, data, isSaveDownload }) {
    _applyStateFromResponse(data);
    _renderState();

    if (runTransformName === "exit") {
      const exitText = (data.output && data.output.trim()) ? data.output : "(no output)";
      appendResultBlock(runTransformName, exitText);
      if (cfg.fields.exit_auto_copy && exitText !== "(no output)") {
        const copied = await copyTextToClipboard(exitText);
        setStatus(
          copied
            ? "Applied exit successfully. Copied plan to clipboard."
            : "Applied exit successfully. Could not copy automatically."
        );
      } else {
        setStatus("Applied exit successfully.");
      }
    } else {
      setStatus("Applied " + runTransformName + " successfully.");
      appendResultBlock(runTransformName, data.output || "");
    }

    if (isSaveDownload) {
      _downloadFromResponse(data);
    }
  }

  async function refresh() {
    try {
      const [transformsRes, stateRes] = await Promise.all([fetch("/api/transforms"), fetch("/api/state")]);
      const transformsData = await transformsRes.json();
      const stateData = await stateRes.json();
      if (transformsData.ok && Array.isArray(transformsData.transforms)) {
        appState.allTransforms = transformsData.transforms;
        if (appState.selectedTransform && !appState.allTransforms.some((item) => item.name === appState.selectedTransform && item.enabled)) {
          appState.selectedTransform = appState.allTransforms.some((item) => item.name === "load" && item.enabled)
            ? "load"
            : (appState.allTransforms.find((item) => item.enabled)?.name || "");
        }
      }
      renderTransforms();
      if (stateData.ok) {
        _applyStateFromResponse(stateData);
      }
      renderModels(appState.latestModels);
      renderInsights();
      updatePanels();
    } catch (err) {
      setStatus("Refresh failed: " + String(err));
      renderTransforms();
      updatePanels();
    }
  }

  function bindLoad() {
    loadBtn.addEventListener("click", async () => {
      if (appState.selectedTransform !== "load") { setStatus("Select load first."); return; }
      const serverPath = (serverPathInput.value || "").trim();
      const file = fileInput.files && fileInput.files[0];
      if (!serverPath && !file) { setStatus("Provide a server path or pick a model file first."); return; }

      loadBtn.disabled = true;
      try {
        const payload = { alias: aliasInput.value || null };
        if (serverPath) {
          payload.server_path = serverPath;
        } else {
          setStatus("Reading file...");
          const bytes = new Uint8Array(await file.arrayBuffer());
          let binary = "";
          const chunk = 0x8000;
          for (let i = 0; i < bytes.length; i += chunk) binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
          payload.filename = file.name;
          payload.content_b64 = btoa(binary);
        }

        setStatus("Loading model via transform...");
        const response = await fetch("/api/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) { setStatus("Load failed: " + (data.error || "unknown error")); return; }
        _applyStateFromResponse(data);
        _renderState();
        setStatus("Load completed successfully.");
      } catch (err) {
        setStatus("Load failed: " + String(err));
      } finally {
        loadBtn.disabled = false;
      }
    });
  }

  function bindRun() {
    transformRunBtn.addEventListener("click", async () => {
      if (!isRunnableTransform(appState.selectedTransform)) { setStatus("Select a READY transform first."); return; }
      const runTransformName = appState.selectedTransform;
      const meta = getTransformMeta(runTransformName);
      const cfg = getTransformConfig(runTransformName);
      const modeKey = typeof meta.mode_key === "string" && meta.mode_key ? meta.mode_key : null;
      const modeNames = Array.isArray(meta.modes) ? meta.modes.map((m) => String(m).toLowerCase()) : [];
      const defaultMode = typeof meta.default_mode === "string" && meta.default_mode
        ? String(meta.default_mode).toLowerCase()
        : (modeNames[0] || "default");
      let selectedMode = null;
      if (modeKey) {
        const rawMode = String(cfg.fields[modeKey] == null ? defaultMode : cfg.fields[modeKey]).trim().toLowerCase();
        if (!modeNames.includes(rawMode)) {
          setStatus("Invalid " + modeKey + " for " + runTransformName + ": " + rawMode);
          return;
        }
        selectedMode = rawMode;
        cfg.fields[modeKey] = selectedMode;
      }
      const allowedByMode = meta.mode_allowed_keys && typeof meta.mode_allowed_keys === "object"
        ? meta.mode_allowed_keys
        : {};
      const requiredByMode = meta.mode_required_keys && typeof meta.mode_required_keys === "object"
        ? meta.mode_required_keys
        : {};
      const allowed = modeKey
        ? (Array.isArray(allowedByMode[selectedMode]) ? allowedByMode[selectedMode].map((k) => String(k)) : [])
        : (Array.isArray(meta.allowed_keys) ? meta.allowed_keys : []);
      const required = new Set(
        modeKey
          ? (Array.isArray(requiredByMode[selectedMode]) ? requiredByMode[selectedMode].map((k) => String(k)) : [])
          : (Array.isArray(meta.required_keys) ? meta.required_keys : [])
      );
      let payload = {};
      for (const key of allowed) {
        const rawValue = cfg.fields[key];
        if (typeof rawValue === "boolean") {
          payload[key] = rawValue;
          continue;
        }
        const parsed = parseFieldValue(rawValue == null ? "" : String(rawValue));
        if (parsed === undefined) {
          if (required.has(key)) { setStatus("Missing required parameter: " + key); return; }
          continue;
        }
        payload[key] = parsed;
      }

      if (runTransformName === "help") {
        const command = String(cfg.fields.help_command || "").trim();
        const subcommand = String(cfg.fields.help_subcommand || "").trim();
        if (!command && subcommand) {
          setStatus("Set command before subcommand for help.");
          return;
        }
        if (!command) {
          payload = {};
        } else if (!subcommand) {
          payload = command;
        } else {
          payload = { [command]: subcommand };
        }
      }

      if (runTransformName === "assert") {
        const yamlExpr = String(cfg.fields.assert_yaml || "").trim();
        if (!yamlExpr) {
          setStatus("Provide an assert YAML expression.");
          return;
        }
        payload = yamlExpr;
      }

      if (runTransformName === "save" && (cfg.save_mode || "server") === "download") {
        payload.format = cfg.save_download_format || "safetensors";
      }

      setStatus("Applying " + runTransformName + "...");
      transformRunBtn.disabled = true;
      progressController.start(runTransformName, getIsIteratingTransform(runTransformName));
      const executeApply = async (previewDecision) => {
        const isSaveDownload = runTransformName === "save" && (cfg.save_mode || "server") === "download";
        const response = await fetch(
          isSaveDownload ? "/api/save_download" : "/api/_apply_transform",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(
              isSaveDownload
                ? { payload: payload }
                : {
                  transform: runTransformName,
                  payload: payload,
                  summary_mode: runTransformName === "exit"
                    ? (String(cfg.fields.exit_summary_mode || "raw").toLowerCase())
                    : undefined,
                  preview_decision: previewDecision,
                }
            ),
          }
        );
        const data = await response.json();
        return { response, data, isSaveDownload };
      };
      try {
        let { response, data, isSaveDownload } = await executeApply(undefined);
        if (
          !isSaveDownload
          && response.ok
          && data.ok
          && data.preview_confirmation_required
          && typeof showPreviewConfirm === "function"
        ) {
          progressController.stop();
          transformRunBtn.disabled = false;
          showPreviewConfirm({
            transformName: runTransformName,
            previewOutput: data.output || "",
            onGo: async () => {
              setStatus("Applying " + runTransformName + " (go)...");
              transformRunBtn.disabled = true;
              progressController.start(runTransformName, getIsIteratingTransform(runTransformName));
              try {
                const goResult = await executeApply("go");
                const goResponse = goResult.response;
                const goData = goResult.data;
                const goIsSaveDownload = goResult.isSaveDownload;
                if (!goResponse.ok || !goData.ok) {
                  setStatus("Apply failed: " + (goData.error || "unknown error"));
                  return;
                }
                await _handleApplySuccess({
                  runTransformName,
                  cfg,
                  data: goData,
                  isSaveDownload: goIsSaveDownload,
                });
              } catch (err) {
                setStatus("Apply failed: " + String(err));
              } finally {
                progressController.stop();
                transformRunBtn.disabled = false;
              }
            },
            onNoGo: () => {
              setStatus("No-go: skipped " + runTransformName + ".");
              appendResultBlock(runTransformName, data.output || "");
            },
          });
          return;
        }
        if (!response.ok || !data.ok) {
          if (
            runTransformName === "assert"
            && data
            && data.error_info
            && data.error_info.code === "assert_error"
            && typeof showAssertFailure === "function"
          ) {
            showAssertFailure(data);
          }
          setStatus("Apply failed: " + (data.error || "unknown error"));
          return;
        }
        await _handleApplySuccess({ runTransformName, cfg, data, isSaveDownload });
      } catch (err) {
        setStatus("Apply failed: " + String(err));
      } finally {
        progressController.stop();
        transformRunBtn.disabled = false;
      }
    });
  }

  return { bindLoad, bindRun, refresh };
}

export { createActions };

// Make the interactive 3D figures fit their output box.
//
// Plotly measures its container once, while the page is still settling, and on these
// pages it reads a width that includes the right-hand contents sidebar -- so the SVG ends
// up exactly one sidebar too wide and the output box grows a horizontal scrollbar. Asking
// Plotly to re-measure once the layout is final, and again whenever the box changes size,
// keeps every figure inside the text column.

(function () {
  function fit() {
    if (!window.Plotly) return;
    document.querySelectorAll(".js-plotly-plot").forEach(function (plot) {
      try {
        window.Plotly.Plots.resize(plot);
      } catch (e) {
        /* a figure that failed to render is not worth breaking the page over */
      }
    });
  }

  function observe() {
    if (!window.ResizeObserver) return;
    var ro = new ResizeObserver(function () {
      fit();
    });
    document.querySelectorAll(".js-plotly-plot").forEach(function (plot) {
      if (plot.parentElement) ro.observe(plot.parentElement);
    });
  }

  function start() {
    fit();
    observe();
    // The sidebar collapse animation resizes the column after the fact.
    window.addEventListener("resize", fit);
    var btn = document.getElementById("pst-collapse-sidebar-button");
    if (btn) btn.addEventListener("click", function () { setTimeout(fit, 500); });
  }

  if (document.readyState === "complete") {
    start();
  } else {
    window.addEventListener("load", start);
  }
})();

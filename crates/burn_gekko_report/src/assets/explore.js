"use strict";
const samples = JSON.parse(document.getElementById("sample-data").textContent);
const selector = document.getElementById("sample");
selector.addEventListener("change", () => {
  const sample = samples[Number(selector.value)];
  if (!sample) return;
  sample.panels.forEach(([title, file], i) => {
    const image = document.getElementById(`panel-${i}`);
    image.src = file;
    image.alt = `${title}, room ${sample.room_seed}, target view ${sample.target_view}`;
  });
  document.getElementById("sample-metric").textContent =
    `MSE ${sample.mse.toFixed(4)} · cosine ${sample.cosine.toFixed(3)}`;
});

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
  const snr = Number.isFinite(sample.feature_snr_db)
    ? `Feature signal/error ${sample.feature_snr_db.toFixed(2)} dB · ` : "";
  document.getElementById("sample-metric").textContent =
    `${snr}MSE ${sample.mse.toFixed(4)} · cosine ${sample.cosine.toFixed(3)}`;
});

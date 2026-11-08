// The model the browser downloads. scripts/export_onnx.py prints bytes and sha256 for a build.
// The hash is checked before the model is loaded, because the file comes from another host.
export const MODEL = {
  url: "https://huggingface.co/Hammy2207/voc2vec-onnx/resolve/main/voc2vec-l12.int8.onnx",
  bytes: 121883139,
  sha256: "8b97cb362ed5445c6a4e6a8132e0238a35fcc8fb5fc23d5b059b8acf04374f9a",
  tag: "voc2vec-l12.int8",
};
export const ORT_VERSION = "1.20.1";

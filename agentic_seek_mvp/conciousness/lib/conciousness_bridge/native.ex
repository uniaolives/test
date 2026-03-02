defmodule ConciousnessBridge.Native do
  use Rustler, otp_app: :conciousness, crate: "rust_sentry", path: "../rust_sentry"

  def get_intent(), do: :erlang.nif_error(:nif_not_loaded)
  def set_intent(_intent), do: :erlang.nif_error(:nif_not_loaded)
  def check_security(_domain), do: :erlang.nif_error(:nif_not_loaded)
  def log_audit(), do: :erlang.nif_error(:nif_not_loaded)
end

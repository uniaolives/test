defmodule ConciousnessWeb do
  use Plug.Router
  import Plug.Conn
  alias ConciousnessBridge.Native

  plug :match
  plug :dispatch

  # --- API HTTP (Para o Python) ---
  get "/status" do
    {:intent, {:current, intent}, {:emotion, emotion}} = Native.get_intent()
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(200, Jason.encode!(%{intent: intent, emotion: emotion}))
  end

  post "/task" do
    {:ok, body, conn} = read_body(conn)
    {:ok, command} = Jason.decode(body)

    # 1. Atualizar Estado Global (Rust NIF)
    Native.set_intent(command["action"])

    # 2. Verificar Segurança (Rust NIF)
    target = Map.get(command, "target", "")
    security = Native.check_security(target)

    if security == :blocked do
      conn
      |> put_resp_content_type("application/json")
      |> send_resp(403, Jason.encode!(%{error: "Security violation"}))
    else
      # 3. Rotear para Go (Conexão via Unix Socket)
      IO.inspect "Enviando comando para Go Worker via Unix Socket..."

      payload = %{action: command["action"], target: target}

      case talk_to_go(payload) do
        {:ok, _resp} ->
          response = %{
            status: "queued",
            target: target,
            processed_by: "Rust_Sentry_Router"
          }

          conn
          |> put_resp_content_type("application/json")
          |> send_resp(200, Jason.encode!(response))
        {:error, reason} ->
          conn
          |> put_resp_content_type("application/json")
          |> send_resp(500, Jason.encode!(%{error: "Go worker unreachable", reason: inspect(reason)}))
      end
    end
  end

  defp talk_to_go(payload) do
    socket_path = "/tmp/atlas_go.sock"
    case :gen_tcp.connect({:local, socket_path}, 0, [:binary, active: false]) do
      {:ok, socket} ->
        :gen_tcp.send(socket, Jason.encode!(payload))
        # No Go code, it decodes one JSON and closes or waits.
        # Let's wait for the response.
        case :gen_tcp.recv(socket, 0, 5000) do
          {:ok, data} ->
            :gen_tcp.close(socket)
            {:ok, Jason.decode!(data)}
          {:error, reason} ->
            :gen_tcp.close(socket)
            {:error, reason}
        end
      {:error, reason} ->
        {:error, reason}
    end
  end

  match _ do
    send_resp(conn, 404, "Not Found")
  end
end

-module(resilience).
-export([loop/1]).

loop(State) ->
    receive
        {agent_crash, _Reason} ->
            % spawn_link(agent_module, init, []),
            loop(State);
        {data, _Payload} ->
            % process(Payload),
            loop(State)
    end.

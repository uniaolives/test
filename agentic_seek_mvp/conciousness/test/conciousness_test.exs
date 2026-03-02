defmodule ConciousnessTest do
  use ExUnit.Case
  doctest Conciousness

  test "greets the world" do
    assert Conciousness.hello() == :world
  end
end

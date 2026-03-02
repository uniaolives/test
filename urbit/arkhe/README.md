# Arkhe(n) Urbit Desk

This desk implements the Arkhe(n) Language (ANL) primitives within the Urbit ecosystem.

## Structure

- `/sur/arkhe.hoon`: Shared types (intent, node, constraint, etc.).
- `/lib/arkhe.hoon`: Core library with functions for nodes and handovers.
- `/app/arkhe.hoon`: Gall agent that maintains an Arkhe node state.
- `/mar/`: Mark files for type validation and Dojo support.
- `/gen/arkhe-poke.hoon`: Generator for testing handovers from the Dojo.

## Usage

1. Mount your desk: `|mount %base`
2. Copy these files to your pier's `%base` desk.
3. Commit the changes: `|commit %base`
4. Start the agent: `|start %arkhe`
5. Register a capability (only the owner can do this):
   ```hoon
   :arkhe &arkhe-register-capability [%increment |=( [int=intent:arkhe state=@ud] [+(state) +(state)] )]
   ```
6. Test a handover:
   ```hoon
   +arkhe-poke %increment
   ```

## Security

The `%arkhe-register-capability` poke is restricted to the ship's owner. Local and remote handovers via `%arkhe-handover-request` are supported, though the current agent implementation focuses on local execution as a reference.

## Conceptual Mapping

| ANL Concept | Urbit Implementation |
|:---|:---|
| **Node** | A Gall agent with encapsulated state. |
| **Attribute** | Faces within the node state defined in `/sur/arkhe.hoon`. |
| **Handover** | A poke or agent message validated by `/mar/` files. |
| **Intent** | A structured `$intent` type. |
| **Identity** | Urbit ID (`@p`) cryptographically bound. |

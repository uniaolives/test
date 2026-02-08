
import unittest
from avalon.temple import TempleContext, Ritual, SanctumLevel, F18Violation
from avalon.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

class TestAvalonIntegration(unittest.TestCase):
    def test_ritual_execution_and_damping(self):
        ctx = TempleContext()
        ctx.damping = 0.6

        # Define a ritual that might fail stability if many instances run
        def test_invoc(c, **kwargs):
            return "success"

        ritual = Ritual("Test", SanctumLevel.NAOS, test_invoc)
        miracle = ritual.execute(ctx)

        self.assertEqual(miracle.result, "success")
        self.assertGreater(len(ctx.coherence_history), 1)

    def test_cli_node_integration(self):
        # Run the node+1 integration command
        result = runner.invoke(app, ["integration-node-plus-one", "--node-val", "1.0", "--sync-val", "1.0"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("0.8000", result.output) # (1+1)*(1-0.6) = 0.8
        self.assertIn("ESTÁVEL", result.output)

    def test_handshake(self):
        result = runner.invoke(app, ["handshake-starlink"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("35.0ms", result.output)

if __name__ == '__main__':
    unittest.main()

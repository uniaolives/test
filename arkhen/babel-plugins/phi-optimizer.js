module.exports = function(babel) {
  const { types: t } = babel;
  return {
    name: "phi-optimizer",
    visitor: {
      Program(path) {
        path.pushContainer('body', t.expressionStatement(t.stringLiteral("ARKHE-PHI-OPTIMIZED")));
      }
    }
  };
};

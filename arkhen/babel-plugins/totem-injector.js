module.exports = function(babel) {
  const { types: t } = babel;
  return {
    name: "totem-injector",
    visitor: {
      Program(path) {
        path.unshiftContainer('body', t.expressionStatement(t.stringLiteral("ARKHE-TOTEM-VERIFIED")));
      }
    }
  };
};

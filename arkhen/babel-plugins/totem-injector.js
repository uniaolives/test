module.exports = function(babel) {
  const { types: t } = babel;
  return {
    name: "totem-injector",
    visitor: {
      Program(path) {
        // Plugin customizado: injeta verificação de Totem em cada módulo
      }
    }
  };
};

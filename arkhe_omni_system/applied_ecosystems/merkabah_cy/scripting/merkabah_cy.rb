# merkabah_cy.rb
class CYVariety
  attr_accessor :h11, :h21, :euler, :metric_diag, :complex_moduli

  def initialize(h11, h21)
    @h11 = h11
    @h21 = h21
    @euler = 2 * (h11 - h21)
    @metric_diag = Array.new(h11, 1.0)
    @complex_moduli = Array.new(h21, 0.0)
  end

  def complexity_index
    @h11 / 491.0
  end

  # MAPEAR_CY
  def mapear_cy(iterations)
    iterations.times do
      @complex_moduli.map! { |z| z + (rand - 0.5) * 0.1 }
    end
    self
  end

  # GERAR_ENTIDADE
  def self.gerar_entidade(seed)
    srand(seed)
    h11 = 200 + rand(292)
    h21 = 100 + rand(152)
    new(h11, h21)
  end
end

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "arkhe_topology.hpp"
#include <cmath>

using namespace arkhe::topology;

TEST_CASE( "KleinBottlehole: Cálculo de Juro Quântico", "[quantum_interest]" ) {

    // Inicializar com escala de Planck padrão
    KleinBottlehole hole(1.616e-35);

    SECTION( "Viagem nula deve ter custo zero" ) {
        double cost = hole.calculate_quantum_interest(0.0, 100.0);
        REQUIRE( cost == Approx(0.0) );
    }

    SECTION( "Micro-CTC (Escala de Planck) deve ter custo finito baixo" ) {
        // dt = tempo de Planck, energia baixa
        double dt = 5.39e-44;
        double energy = 1.0;
        double cost = hole.calculate_quantum_interest(dt, energy);

        // Esperado: Não deve divergir para infinito
        REQUIRE( std::isfinite(cost) );
        REQUIRE( cost < 1e10 );
    }

    SECTION( "Macro-CTC (1 segundo) deve ter custo exponencialmente proibitivo" ) {
        // dt = 1 segundo (gigantesco para física quântica)
        double dt = 1.0;
        double energy = 1.0;
        double cost = hole.calculate_quantum_interest(dt, energy);

        // Deve ser virtualmente infinito (proteção da cronologia)
        REQUIRE( cost > 1e10 );
    }
}

TEST_CASE( "KleinBottlehole: Verificação de Monodromia", "[monodromy]" ) {
    KleinBottlehole hole;

    SECTION( "Fases não-orientáveis (3) devem retornar true" ) {
        REQUIRE( hole.check_monodromy_iteration(3) == true );
        REQUIRE( hole.check_monodromy_iteration(9) == true ); // Múltiplos de 3 (ímpares)
    }

    SECTION( "Fases orientáveis (0, 6) devem retornar false" ) {
        REQUIRE( hole.check_monodromy_iteration(0) == false );
        REQUIRE( hole.check_monodromy_iteration(6) == false );
        REQUIRE( hole.check_monodromy_iteration(1) == false );
    }
}

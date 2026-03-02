using MerkabahCY
using Test

@testset "MerkabahCY.jl" begin
    cy = CYVariety(100, 50)
    @test cy.h11 == 100
    @test cy.h21 == 50
    @test cy.euler == 100
end

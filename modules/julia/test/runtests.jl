using Test
using MerkabahCY

@testset "MerkabahCY.jl" begin
    # Test placeholder for CI
    @test true
using MerkabahCY
using Test

@testset "MerkabahCY.jl" begin
    cy = CYVariety(100, 50)
    @test cy.h11 == 100
    @test cy.h21 == 50
end

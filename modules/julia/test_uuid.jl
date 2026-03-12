using Pkg
for (uuid, pkg) in Pkg.dependencies()
    if pkg.name == "Test"
        println("NAME: ", pkg.name)
        println("UUID: ", uuid)
    end
end

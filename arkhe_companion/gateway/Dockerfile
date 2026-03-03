FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src
COPY ["Arkhe.Gateway/Arkhe.Gateway.csproj", "Arkhe.Gateway/"]
COPY ["../proto/arkhe.proto", "proto/"]
RUN dotnet restore "Arkhe.Gateway/Arkhe.Gateway.csproj"
COPY . .
WORKDIR "/src/Arkhe.Gateway"
RUN dotnet build "Arkhe.Gateway.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "Arkhe.Gateway.csproj" -c Release -o /app/publish

FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "Arkhe.Gateway.dll"]

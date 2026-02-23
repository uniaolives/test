# TelegramConstellate: A comprehensive tool for networking Telegram data

Developed by [Isabela Rocha](https://github.com/rocha-isabela/) (Brazil), TelegramConstellate builds co-occurrence networks (or constellations) from Telegram data. It applies SimHash clustering to identify messages that are identical or nearly identical, even if small variations exist (such as different links or minor text edits).

## Methodology
Nodes (Groups or Authors) are linked whenever they contain messages with the same SimHashID, revealing hidden connections of content circulation across the platform. This methodology ensures that identical or near-identical messages are grouped, enabling the study of narrative propagation, cross-group message diffusion, and coordinated behavior.

## How to cite this code
Rocha, Isabela, Dashichev, Aleksandr. (2025) TelegramConstellate: A comprehensive tool for networking Telegram data. Pre-print. Available at https://doi.org/10.5281/zenodo.17459043.

## Integration with SASC
The SASC implementation of this logic is available via the `sasc-constellate` tool and the `aletheia_simhash` pattern engine. It leverages SimHash clustering for multidimensional active metadata analysis (Spatial, Temporal, Spectral, Topological) for reality verification and deepfake detection.

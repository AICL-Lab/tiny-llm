## MODIFIED Requirements

### Requirement: Model Loading

The system SHALL parse and load model files with explicit validation of malformed, incomplete, or unsupported GGUF inputs before runtime allocation begins.

1. The system SHALL parse GGUF file headers and configuration metadata.
2. The system SHALL reject GGUF metadata, tensor dimensions, or array sizes that overflow safe allocation math.
3. The system SHALL validate that required runtime tensors are present before allocating CUDA memory for GGUF runtime loading.
4. The system SHALL return a structured error for corrupted, invalid, incomplete, or unsupported model files.
5. The system SHALL provide a structured model representation including `ModelConfig` and `QuantizedWeight` structures.

#### Scenario: Reject oversized GGUF tensor metadata
- **GIVEN** a GGUF tensor descriptor or metadata array whose size computation would overflow
- **WHEN** the parser reads the file
- **THEN** parsing SHALL fail with a structured error
- **AND** the parser SHALL NOT attempt an undersized allocation

#### Scenario: Reject incomplete GGUF runtime tensor sets
- **GIVEN** a GGUF file that parses successfully but omits tensors required for runtime inference
- **WHEN** `ModelLoader::loadGGUF()` is called
- **THEN** the loader SHALL return an error describing the missing or unsupported tensors
- **AND** it SHALL fail before starting CUDA allocations for model weights

#### Scenario: Ignore unrelated metadata fallbacks
- **GIVEN** GGUF metadata that includes unrelated keys such as `general.architecture`
- **WHEN** `extractModelConfig()` maps metadata into `ModelConfig`
- **THEN** unrelated keys SHALL NOT override numeric model dimensions

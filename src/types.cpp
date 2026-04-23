#include "tiny_llm/types.h"
#include "tiny_llm/validator.h"

namespace tiny_llm {

Result<void> GenerationConfig::validate() const {
    return Validator::validateGenerationConfig(*this);
}

} // namespace tiny_llm

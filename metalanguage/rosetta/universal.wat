(module
  (func $check_threshold (param $phi f32) (result i32)
    local.get $phi
    f32.const 0.618
    f32.gt
  )
)

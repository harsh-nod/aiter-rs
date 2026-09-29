use fe2o3_kernel_ir::*;

fn operation(id: u32, ty: Type, kind: OperationKind) -> Operation {
    Operation::effect_free(ValueDef::new(ValueId(id), ty), kind)
}

fn wave_shuffle_module(shuffle_in_even_branch: bool, active_lanes: u32) -> Module {
    let u32_type = Type::Scalar(ScalarType::U32);
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations = vec![
        operation(
            0,
            u32_type.clone(),
            OperationKind::Wave(WaveOperation::full(
                WaveOperationKind::LaneId,
                WaveWidth::Wave64,
            )),
        ),
        operation(1, u32_type.clone(), OperationKind::Constant(Constant::U32(1))),
        operation(
            2,
            u32_type.clone(),
            OperationKind::Binary {
                op: BinaryOp::BitAnd,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
        ),
        operation(3, u32_type.clone(), OperationKind::Constant(Constant::U32(0))),
        operation(
            4,
            Type::BOOL,
            OperationKind::Compare {
                predicate: ComparePredicate::Equal,
                lhs: ValueId(2),
                rhs: ValueId(3),
            },
        ),
        operation(
            5,
            u32_type.clone(),
            OperationKind::Binary {
                op: BinaryOp::Add,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
        ),
    ];

    let shuffle = operation(
        6,
        u32_type.clone(),
        OperationKind::Wave(WaveOperation {
            kind: WaveOperationKind::ShuffleIndex {
                value: ValueId(0),
                source_lane: ValueId(5),
                tile_width: 64,
            },
            width: WaveWidth::Wave64,
            active_lanes,
            convergence: Convergence::uniform(SynchronizationScope::Subgroup),
        }),
    );
    if !shuffle_in_even_branch {
        entry.operations.push(shuffle.clone());
    }
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(4),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });

    let mut even = BasicBlock::new(BlockId(1));
    if shuffle_in_even_branch {
        even.operations.push(shuffle);
    }
    even.terminator = Some(Terminator::Return {
        values: vec![ValueId(6)],
    });

    let mut odd = BasicBlock::new(BlockId(2));
    odd.terminator = Some(Terminator::Return {
        values: vec![ValueId(0)],
    });

    let mut function = Function::definition(
        "shuffle_participation",
        Signature::new(vec![], vec![u32_type]),
        vec![],
        vec![entry, even, odd],
    );
    function
        .required_capabilities
        .insert(TargetCapability::WaveWidth(WaveWidth::Wave64));
    let mut module = Module::new("aiter_rs::wave_participation");
    module.functions.push(function);
    module
}

#[test]
fn declared_full_wave_in_lane_varying_branch_is_currently_accepted() {
    // The odd source lanes skip this operation, despite the full-wave claim.
    let module = wave_shuffle_module(true, 64);
    assert_eq!(verify_module(&module), Ok(()));
}

#[test]
fn explicit_partial_wave_is_rejected() {
    let module = wave_shuffle_module(true, 32);
    let errors = verify_module(&module).expect_err("partial wave must be rejected");
    assert!(errors.contains(DiagnosticCode::InvalidWaveOperation));
}

#[test]
fn moving_shuffle_before_lane_varying_branch_is_accepted() {
    let module = wave_shuffle_module(false, 64);
    assert_eq!(verify_module(&module), Ok(()));
}


A
input_1Placeholder*
dtype0*
shape:’’’’’’’’’
Q
dense_1/random_uniform/shapeConst*
valueB"   d   *
dtype0
G
dense_1/random_uniform/minConst*
valueB
 *B[x¾*
dtype0
G
dense_1/random_uniform/maxConst*
valueB
 *B[x>*
dtype0

$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
T0*
dtype0*
seed2¢ÆĆ*
seed±’å)
b
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0
l
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0
^
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0
b
dense_1/kernel
VariableV2*
dtype0*
	container *
shape
:d*
shared_name 

dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(
[
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel
>
dense_1/ConstConst*
valueBd*    *
dtype0
\
dense_1/bias
VariableV2*
	container *
shape:d*
shared_name *
dtype0

dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(
U
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias
e
dense_1/MatMulMatMulinput_1dense_1/kernel/read*
T0*
transpose_a( *
transpose_b( 
]
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*
T0
.
dense_1/ReluReludense_1/BiasAdd*
T0
X
#res1a_branch2a/random_uniform/shapeConst*
valueB"d   d   *
dtype0
N
!res1a_branch2a/random_uniform/minConst*
dtype0*
valueB
 *¬\1¾
N
!res1a_branch2a/random_uniform/maxConst*
dtype0*
valueB
 *¬\1>

+res1a_branch2a/random_uniform/RandomUniformRandomUniform#res1a_branch2a/random_uniform/shape*
dtype0*
seed2īĪ*
seed±’å)*
T0
w
!res1a_branch2a/random_uniform/subSub!res1a_branch2a/random_uniform/max!res1a_branch2a/random_uniform/min*
T0

!res1a_branch2a/random_uniform/mulMul+res1a_branch2a/random_uniform/RandomUniform!res1a_branch2a/random_uniform/sub*
T0
s
res1a_branch2a/random_uniformAdd!res1a_branch2a/random_uniform/mul!res1a_branch2a/random_uniform/min*
T0
i
res1a_branch2a/kernel
VariableV2*
shape
:dd*
shared_name *
dtype0*
	container 
ø
res1a_branch2a/kernel/AssignAssignres1a_branch2a/kernelres1a_branch2a/random_uniform*
T0*(
_class
loc:@res1a_branch2a/kernel*
validate_shape(*
use_locking(
p
res1a_branch2a/kernel/readIdentityres1a_branch2a/kernel*
T0*(
_class
loc:@res1a_branch2a/kernel
E
res1a_branch2a/ConstConst*
valueBd*    *
dtype0
c
res1a_branch2a/bias
VariableV2*
shape:d*
shared_name *
dtype0*
	container 
©
res1a_branch2a/bias/AssignAssignres1a_branch2a/biasres1a_branch2a/Const*
use_locking(*
T0*&
_class
loc:@res1a_branch2a/bias*
validate_shape(
j
res1a_branch2a/bias/readIdentityres1a_branch2a/bias*
T0*&
_class
loc:@res1a_branch2a/bias
x
res1a_branch2a/MatMulMatMuldense_1/Relures1a_branch2a/kernel/read*
T0*
transpose_a( *
transpose_b( 
r
res1a_branch2a/BiasAddBiasAddres1a_branch2a/MatMulres1a_branch2a/bias/read*
data_formatNHWC*
T0
:
activation_1/ReluRelures1a_branch2a/BiasAdd*
T0
X
#res1a_branch2b/random_uniform/shapeConst*
valueB"d   d   *
dtype0
N
!res1a_branch2b/random_uniform/minConst*
valueB
 *¬\1¾*
dtype0
N
!res1a_branch2b/random_uniform/maxConst*
dtype0*
valueB
 *¬\1>

+res1a_branch2b/random_uniform/RandomUniformRandomUniform#res1a_branch2b/random_uniform/shape*
seed±’å)*
T0*
dtype0*
seed2ć¤
w
!res1a_branch2b/random_uniform/subSub!res1a_branch2b/random_uniform/max!res1a_branch2b/random_uniform/min*
T0

!res1a_branch2b/random_uniform/mulMul+res1a_branch2b/random_uniform/RandomUniform!res1a_branch2b/random_uniform/sub*
T0
s
res1a_branch2b/random_uniformAdd!res1a_branch2b/random_uniform/mul!res1a_branch2b/random_uniform/min*
T0
i
res1a_branch2b/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape
:dd
ø
res1a_branch2b/kernel/AssignAssignres1a_branch2b/kernelres1a_branch2b/random_uniform*
use_locking(*
T0*(
_class
loc:@res1a_branch2b/kernel*
validate_shape(
p
res1a_branch2b/kernel/readIdentityres1a_branch2b/kernel*
T0*(
_class
loc:@res1a_branch2b/kernel
E
res1a_branch2b/ConstConst*
valueBd*    *
dtype0
c
res1a_branch2b/bias
VariableV2*
dtype0*
	container *
shape:d*
shared_name 
©
res1a_branch2b/bias/AssignAssignres1a_branch2b/biasres1a_branch2b/Const*
use_locking(*
T0*&
_class
loc:@res1a_branch2b/bias*
validate_shape(
j
res1a_branch2b/bias/readIdentityres1a_branch2b/bias*&
_class
loc:@res1a_branch2b/bias*
T0
}
res1a_branch2b/MatMulMatMulactivation_1/Relures1a_branch2b/kernel/read*
T0*
transpose_a( *
transpose_b( 
r
res1a_branch2b/BiasAddBiasAddres1a_branch2b/MatMulres1a_branch2b/bias/read*
T0*
data_formatNHWC
?
	add_1/addAddres1a_branch2b/BiasAdddense_1/Relu*
T0
-
activation_2/ReluRelu	add_1/add*
T0
X
#res1b_branch2a/random_uniform/shapeConst*
valueB"d   d   *
dtype0
N
!res1b_branch2a/random_uniform/minConst*
valueB
 *¬\1¾*
dtype0
N
!res1b_branch2a/random_uniform/maxConst*
valueB
 *¬\1>*
dtype0

+res1b_branch2a/random_uniform/RandomUniformRandomUniform#res1b_branch2a/random_uniform/shape*
dtype0*
seed2ń§N*
seed±’å)*
T0
w
!res1b_branch2a/random_uniform/subSub!res1b_branch2a/random_uniform/max!res1b_branch2a/random_uniform/min*
T0

!res1b_branch2a/random_uniform/mulMul+res1b_branch2a/random_uniform/RandomUniform!res1b_branch2a/random_uniform/sub*
T0
s
res1b_branch2a/random_uniformAdd!res1b_branch2a/random_uniform/mul!res1b_branch2a/random_uniform/min*
T0
i
res1b_branch2a/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape
:dd
ø
res1b_branch2a/kernel/AssignAssignres1b_branch2a/kernelres1b_branch2a/random_uniform*
use_locking(*
T0*(
_class
loc:@res1b_branch2a/kernel*
validate_shape(
p
res1b_branch2a/kernel/readIdentityres1b_branch2a/kernel*
T0*(
_class
loc:@res1b_branch2a/kernel
E
res1b_branch2a/ConstConst*
valueBd*    *
dtype0
c
res1b_branch2a/bias
VariableV2*
	container *
shape:d*
shared_name *
dtype0
©
res1b_branch2a/bias/AssignAssignres1b_branch2a/biasres1b_branch2a/Const*
T0*&
_class
loc:@res1b_branch2a/bias*
validate_shape(*
use_locking(
j
res1b_branch2a/bias/readIdentityres1b_branch2a/bias*
T0*&
_class
loc:@res1b_branch2a/bias
}
res1b_branch2a/MatMulMatMulactivation_2/Relures1b_branch2a/kernel/read*
T0*
transpose_a( *
transpose_b( 
r
res1b_branch2a/BiasAddBiasAddres1b_branch2a/MatMulres1b_branch2a/bias/read*
T0*
data_formatNHWC
:
activation_3/ReluRelures1b_branch2a/BiasAdd*
T0
X
#res1b_branch2b/random_uniform/shapeConst*
valueB"d   d   *
dtype0
N
!res1b_branch2b/random_uniform/minConst*
valueB
 *¬\1¾*
dtype0
N
!res1b_branch2b/random_uniform/maxConst*
valueB
 *¬\1>*
dtype0

+res1b_branch2b/random_uniform/RandomUniformRandomUniform#res1b_branch2b/random_uniform/shape*
seed±’å)*
T0*
dtype0*
seed2¤ø
w
!res1b_branch2b/random_uniform/subSub!res1b_branch2b/random_uniform/max!res1b_branch2b/random_uniform/min*
T0

!res1b_branch2b/random_uniform/mulMul+res1b_branch2b/random_uniform/RandomUniform!res1b_branch2b/random_uniform/sub*
T0
s
res1b_branch2b/random_uniformAdd!res1b_branch2b/random_uniform/mul!res1b_branch2b/random_uniform/min*
T0
i
res1b_branch2b/kernel
VariableV2*
dtype0*
	container *
shape
:dd*
shared_name 
ø
res1b_branch2b/kernel/AssignAssignres1b_branch2b/kernelres1b_branch2b/random_uniform*
use_locking(*
T0*(
_class
loc:@res1b_branch2b/kernel*
validate_shape(
p
res1b_branch2b/kernel/readIdentityres1b_branch2b/kernel*
T0*(
_class
loc:@res1b_branch2b/kernel
E
res1b_branch2b/ConstConst*
valueBd*    *
dtype0
c
res1b_branch2b/bias
VariableV2*
dtype0*
	container *
shape:d*
shared_name 
©
res1b_branch2b/bias/AssignAssignres1b_branch2b/biasres1b_branch2b/Const*
use_locking(*
T0*&
_class
loc:@res1b_branch2b/bias*
validate_shape(
j
res1b_branch2b/bias/readIdentityres1b_branch2b/bias*
T0*&
_class
loc:@res1b_branch2b/bias
}
res1b_branch2b/MatMulMatMulactivation_3/Relures1b_branch2b/kernel/read*
T0*
transpose_a( *
transpose_b( 
r
res1b_branch2b/BiasAddBiasAddres1b_branch2b/MatMulres1b_branch2b/bias/read*
data_formatNHWC*
T0
D
	add_2/addAddres1b_branch2b/BiasAddactivation_2/Relu*
T0
-
activation_4/ReluRelu	add_2/add*
T0
Q
dense_2/random_uniform/shapeConst*
dtype0*
valueB"d      
G
dense_2/random_uniform/minConst*
dtype0*
valueB
 *ž{r¾
G
dense_2/random_uniform/maxConst*
valueB
 *ž{r>*
dtype0

$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0*
seed2”*
seed±’å)
b
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0
l
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0
^
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0
b
dense_2/kernel
VariableV2*
dtype0*
	container *
shape
:d*
shared_name 

dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(
[
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel
>
dense_2/ConstConst*
valueB*    *
dtype0
\
dense_2/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name 

dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
_class
loc:@dense_2/bias*
validate_shape(*
use_locking(*
T0
U
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias
o
dense_2/MatMulMatMulactivation_4/Reludense_2/kernel/read*
transpose_a( *
transpose_b( *
T0
]
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*
T0
E
iterations/initial_valueConst*
valueB
 *    *
dtype0
V

iterations
VariableV2*
shared_name *
dtype0*
	container *
shape: 

iterations/AssignAssign
iterationsiterations/initial_value*
use_locking(*
T0*
_class
loc:@iterations*
validate_shape(
O
iterations/readIdentity
iterations*
T0*
_class
loc:@iterations
=
lr/initial_valueConst*
valueB
 *o:*
dtype0
N
lr
VariableV2*
shared_name *
dtype0*
	container *
shape: 
r
	lr/AssignAssignlrlr/initial_value*
use_locking(*
T0*
_class
	loc:@lr*
validate_shape(
7
lr/readIdentitylr*
T0*
_class
	loc:@lr
A
beta_1/initial_valueConst*
valueB
 *fff?*
dtype0
R
beta_1
VariableV2*
dtype0*
	container *
shape: *
shared_name 

beta_1/AssignAssignbeta_1beta_1/initial_value*
use_locking(*
T0*
_class
loc:@beta_1*
validate_shape(
C
beta_1/readIdentitybeta_1*
T0*
_class
loc:@beta_1
A
beta_2/initial_valueConst*
valueB
 *w¾?*
dtype0
R
beta_2
VariableV2*
shape: *
shared_name *
dtype0*
	container 

beta_2/AssignAssignbeta_2beta_2/initial_value*
_class
loc:@beta_2*
validate_shape(*
use_locking(*
T0
C
beta_2/readIdentitybeta_2*
T0*
_class
loc:@beta_2
@
decay/initial_valueConst*
valueB
 *    *
dtype0
Q
decay
VariableV2*
shape: *
shared_name *
dtype0*
	container 
~
decay/AssignAssigndecaydecay/initial_value*
T0*
_class

loc:@decay*
validate_shape(*
use_locking(
@

decay/readIdentitydecay*
T0*
_class

loc:@decay
L
dense_2_sample_weightsPlaceholder*
dtype0*
shape:’’’’’’’’’
Q
dense_2_targetPlaceholder*%
shape:’’’’’’’’’’’’’’’’’’*
dtype0
4
subSubdense_2/BiasAdddense_2_target*
T0

SquareSquaresub*
T0
@
Mean/reduction_indicesConst*
value	B :*
dtype0
R
MeanMeanSquareMean/reduction_indices*

Tidx0*
	keep_dims( *
T0
A
Mean_1/reduction_indicesConst*
valueB *
dtype0
T
Mean_1MeanMeanMean_1/reduction_indices*

Tidx0*
	keep_dims( *
T0
3
mulMulMean_1dense_2_sample_weights*
T0
7

NotEqual/yConst*
valueB
 *    *
dtype0
A
NotEqualNotEqualdense_2_sample_weights
NotEqual/y*
T0
>
CastCastNotEqual*

SrcT0
*
Truncate( *

DstT0
3
ConstConst*
valueB: *
dtype0
A
Mean_2MeanCastConst*
T0*

Tidx0*
	keep_dims( 
(
truedivRealDivmulMean_2*
T0
5
Const_1Const*
valueB: *
dtype0
F
Mean_3MeantruedivConst_1*
T0*

Tidx0*
	keep_dims( 
4
mul_1/xConst*
dtype0*
valueB
 *  ?
&
mul_1Mulmul_1/xMean_3*
T0
:
ArgMax/dimensionConst*
value	B :*
dtype0
Z
ArgMaxArgMaxdense_2_targetArgMax/dimension*
output_type0	*

Tidx0*
T0
<
ArgMax_1/dimensionConst*
value	B :*
dtype0
_
ArgMax_1ArgMaxdense_2/BiasAddArgMax_1/dimension*
output_type0	*

Tidx0*
T0
)
EqualEqualArgMaxArgMax_1*
T0	
=
Cast_1CastEqual*

DstT0*

SrcT0
*
Truncate( 
5
Const_2Const*
valueB: *
dtype0
E
Mean_4MeanCast_1Const_2*
T0*

Tidx0*
	keep_dims( 
#

group_depsNoOp^Mean_4^mul_1
R
gradients/ShapeConst*
valueB *
_class

loc:@mul_1*
dtype0
Z
gradients/grad_ys_0Const*
valueB
 *  ?*
_class

loc:@mul_1*
dtype0
q
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_class

loc:@mul_1
Z
gradients/mul_1_grad/MulMulgradients/FillMean_3*
T0*
_class

loc:@mul_1
]
gradients/mul_1_grad/Mul_1Mulgradients/Fillmul_1/x*
T0*
_class

loc:@mul_1
l
#gradients/Mean_3_grad/Reshape/shapeConst*
valueB:*
_class
loc:@Mean_3*
dtype0

gradients/Mean_3_grad/ReshapeReshapegradients/mul_1_grad/Mul_1#gradients/Mean_3_grad/Reshape/shape*
T0*
Tshape0*
_class
loc:@Mean_3
a
gradients/Mean_3_grad/ShapeShapetruediv*
T0*
out_type0*
_class
loc:@Mean_3

gradients/Mean_3_grad/TileTilegradients/Mean_3_grad/Reshapegradients/Mean_3_grad/Shape*
_class
loc:@Mean_3*

Tmultiples0*
T0
c
gradients/Mean_3_grad/Shape_1Shapetruediv*
T0*
out_type0*
_class
loc:@Mean_3
a
gradients/Mean_3_grad/Shape_2Const*
valueB *
_class
loc:@Mean_3*
dtype0
d
gradients/Mean_3_grad/ConstConst*
valueB: *
_class
loc:@Mean_3*
dtype0

gradients/Mean_3_grad/ProdProdgradients/Mean_3_grad/Shape_1gradients/Mean_3_grad/Const*

Tidx0*
	keep_dims( *
T0*
_class
loc:@Mean_3
f
gradients/Mean_3_grad/Const_1Const*
valueB: *
_class
loc:@Mean_3*
dtype0
£
gradients/Mean_3_grad/Prod_1Prodgradients/Mean_3_grad/Shape_2gradients/Mean_3_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_class
loc:@Mean_3
d
gradients/Mean_3_grad/Maximum/yConst*
dtype0*
value	B :*
_class
loc:@Mean_3

gradients/Mean_3_grad/MaximumMaximumgradients/Mean_3_grad/Prod_1gradients/Mean_3_grad/Maximum/y*
T0*
_class
loc:@Mean_3

gradients/Mean_3_grad/floordivFloorDivgradients/Mean_3_grad/Prodgradients/Mean_3_grad/Maximum*
_class
loc:@Mean_3*
T0

gradients/Mean_3_grad/CastCastgradients/Mean_3_grad/floordiv*
Truncate( *

DstT0*

SrcT0*
_class
loc:@Mean_3

gradients/Mean_3_grad/truedivRealDivgradients/Mean_3_grad/Tilegradients/Mean_3_grad/Cast*
T0*
_class
loc:@Mean_3
_
gradients/truediv_grad/ShapeShapemul*
out_type0*
_class
loc:@truediv*
T0
c
gradients/truediv_grad/Shape_1Const*
valueB *
_class
loc:@truediv*
dtype0
Ø
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*
T0*
_class
loc:@truediv
u
gradients/truediv_grad/RealDivRealDivgradients/Mean_3_grad/truedivMean_2*
T0*
_class
loc:@truediv
±
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_class
loc:@truediv

gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
T0*
Tshape0*
_class
loc:@truediv
K
gradients/truediv_grad/NegNegmul*
T0*
_class
loc:@truediv
t
 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/NegMean_2*
T0*
_class
loc:@truediv
z
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1Mean_2*
T0*
_class
loc:@truediv

gradients/truediv_grad/mulMulgradients/Mean_3_grad/truediv gradients/truediv_grad/RealDiv_2*
T0*
_class
loc:@truediv
±
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_class
loc:@truediv

 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
T0*
Tshape0*
_class
loc:@truediv
Z
gradients/mul_grad/ShapeShapeMean_1*
T0*
out_type0*
_class

loc:@mul
l
gradients/mul_grad/Shape_1Shapedense_2_sample_weights*
T0*
out_type0*
_class

loc:@mul

(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*
_class

loc:@mul
v
gradients/mul_grad/MulMulgradients/truediv_grad/Reshapedense_2_sample_weights*
T0*
_class

loc:@mul

gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
T0*
_class

loc:@mul*

Tidx0*
	keep_dims( 

gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*
_class

loc:@mul
h
gradients/mul_grad/Mul_1MulMean_1gradients/truediv_grad/Reshape*
T0*
_class

loc:@mul
£
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_class

loc:@mul

gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*
_class

loc:@mul
^
gradients/Mean_1_grad/ShapeShapeMean*
T0*
out_type0*
_class
loc:@Mean_1
_
gradients/Mean_1_grad/SizeConst*
value	B :*
_class
loc:@Mean_1*
dtype0
z
gradients/Mean_1_grad/addAddMean_1/reduction_indicesgradients/Mean_1_grad/Size*
T0*
_class
loc:@Mean_1

gradients/Mean_1_grad/modFloorModgradients/Mean_1_grad/addgradients/Mean_1_grad/Size*
T0*
_class
loc:@Mean_1
f
gradients/Mean_1_grad/Shape_1Const*
dtype0*
valueB: *
_class
loc:@Mean_1
f
!gradients/Mean_1_grad/range/startConst*
value	B : *
_class
loc:@Mean_1*
dtype0
f
!gradients/Mean_1_grad/range/deltaConst*
value	B :*
_class
loc:@Mean_1*
dtype0
­
gradients/Mean_1_grad/rangeRange!gradients/Mean_1_grad/range/startgradients/Mean_1_grad/Size!gradients/Mean_1_grad/range/delta*
_class
loc:@Mean_1*

Tidx0
e
 gradients/Mean_1_grad/Fill/valueConst*
value	B :*
_class
loc:@Mean_1*
dtype0

gradients/Mean_1_grad/FillFillgradients/Mean_1_grad/Shape_1 gradients/Mean_1_grad/Fill/value*
T0*

index_type0*
_class
loc:@Mean_1
Ņ
#gradients/Mean_1_grad/DynamicStitchDynamicStitchgradients/Mean_1_grad/rangegradients/Mean_1_grad/modgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Fill*
T0*
_class
loc:@Mean_1*
N
d
gradients/Mean_1_grad/Maximum/yConst*
value	B :*
_class
loc:@Mean_1*
dtype0

gradients/Mean_1_grad/MaximumMaximum#gradients/Mean_1_grad/DynamicStitchgradients/Mean_1_grad/Maximum/y*
T0*
_class
loc:@Mean_1

gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Maximum*
T0*
_class
loc:@Mean_1

gradients/Mean_1_grad/ReshapeReshapegradients/mul_grad/Reshape#gradients/Mean_1_grad/DynamicStitch*
T0*
Tshape0*
_class
loc:@Mean_1

gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/floordiv*
T0*
_class
loc:@Mean_1*

Tmultiples0
`
gradients/Mean_1_grad/Shape_2ShapeMean*
T0*
out_type0*
_class
loc:@Mean_1
b
gradients/Mean_1_grad/Shape_3ShapeMean_1*
T0*
out_type0*
_class
loc:@Mean_1
d
gradients/Mean_1_grad/ConstConst*
valueB: *
_class
loc:@Mean_1*
dtype0

gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const*
T0*
_class
loc:@Mean_1*

Tidx0*
	keep_dims( 
f
gradients/Mean_1_grad/Const_1Const*
dtype0*
valueB: *
_class
loc:@Mean_1
£
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_3gradients/Mean_1_grad/Const_1*
T0*
_class
loc:@Mean_1*

Tidx0*
	keep_dims( 
f
!gradients/Mean_1_grad/Maximum_1/yConst*
value	B :*
_class
loc:@Mean_1*
dtype0

gradients/Mean_1_grad/Maximum_1Maximumgradients/Mean_1_grad/Prod_1!gradients/Mean_1_grad/Maximum_1/y*
T0*
_class
loc:@Mean_1

 gradients/Mean_1_grad/floordiv_1FloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum_1*
T0*
_class
loc:@Mean_1

gradients/Mean_1_grad/CastCast gradients/Mean_1_grad/floordiv_1*

SrcT0*
_class
loc:@Mean_1*
Truncate( *

DstT0

gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*
_class
loc:@Mean_1
\
gradients/Mean_grad/ShapeShapeSquare*
T0*
out_type0*
_class
	loc:@Mean
[
gradients/Mean_grad/SizeConst*
value	B :*
_class
	loc:@Mean*
dtype0
r
gradients/Mean_grad/addAddMean/reduction_indicesgradients/Mean_grad/Size*
T0*
_class
	loc:@Mean
x
gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
T0*
_class
	loc:@Mean
]
gradients/Mean_grad/Shape_1Const*
dtype0*
valueB *
_class
	loc:@Mean
b
gradients/Mean_grad/range/startConst*
value	B : *
_class
	loc:@Mean*
dtype0
b
gradients/Mean_grad/range/deltaConst*
dtype0*
value	B :*
_class
	loc:@Mean
£
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*

Tidx0*
_class
	loc:@Mean
a
gradients/Mean_grad/Fill/valueConst*
value	B :*
_class
	loc:@Mean*
dtype0

gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
T0*

index_type0*
_class
	loc:@Mean
Ę
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
T0*
_class
	loc:@Mean*
N
`
gradients/Mean_grad/Maximum/yConst*
value	B :*
_class
	loc:@Mean*
dtype0

gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
T0*
_class
	loc:@Mean

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
_class
	loc:@Mean*
T0

gradients/Mean_grad/ReshapeReshapegradients/Mean_1_grad/truediv!gradients/Mean_grad/DynamicStitch*
Tshape0*
_class
	loc:@Mean*
T0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*

Tmultiples0*
T0*
_class
	loc:@Mean
^
gradients/Mean_grad/Shape_2ShapeSquare*
T0*
out_type0*
_class
	loc:@Mean
\
gradients/Mean_grad/Shape_3ShapeMean*
T0*
out_type0*
_class
	loc:@Mean
`
gradients/Mean_grad/ConstConst*
valueB: *
_class
	loc:@Mean*
dtype0

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
T0*
_class
	loc:@Mean*

Tidx0*
	keep_dims( 
b
gradients/Mean_grad/Const_1Const*
valueB: *
_class
	loc:@Mean*
dtype0

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_class
	loc:@Mean
b
gradients/Mean_grad/Maximum_1/yConst*
value	B :*
_class
	loc:@Mean*
dtype0

gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
T0*
_class
	loc:@Mean

gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
T0*
_class
	loc:@Mean

gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*
_class
	loc:@Mean*
Truncate( *

DstT0*

SrcT0
|
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*
_class
	loc:@Mean

gradients/Square_grad/ConstConst^gradients/Mean_grad/truediv*
valueB
 *   @*
_class
loc:@Square*
dtype0
f
gradients/Square_grad/MulMulsubgradients/Square_grad/Const*
T0*
_class
loc:@Square
~
gradients/Square_grad/Mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/Mul*
T0*
_class
loc:@Square
c
gradients/sub_grad/ShapeShapedense_2/BiasAdd*
T0*
out_type0*
_class

loc:@sub
d
gradients/sub_grad/Shape_1Shapedense_2_target*
T0*
out_type0*
_class

loc:@sub

(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*
_class

loc:@sub
¢
gradients/sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_class

loc:@sub

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*
_class

loc:@sub
¦
gradients/sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_class

loc:@sub
X
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_class

loc:@sub

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
Tshape0*
_class

loc:@sub*
T0

*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/sub_grad/Reshape*
data_formatNHWC*
T0*"
_class
loc:@dense_2/BiasAdd
±
$gradients/dense_2/MatMul_grad/MatMulMatMulgradients/sub_grad/Reshapedense_2/kernel/read*
T0*!
_class
loc:@dense_2/MatMul*
transpose_a( *
transpose_b(
±
&gradients/dense_2/MatMul_grad/MatMul_1MatMulactivation_4/Relugradients/sub_grad/Reshape*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
transpose_a(

)gradients/activation_4/Relu_grad/ReluGradReluGrad$gradients/dense_2/MatMul_grad/MatMulactivation_4/Relu*
T0*$
_class
loc:@activation_4/Relu
v
gradients/add_2/add_grad/ShapeShaperes1b_branch2b/BiasAdd*
T0*
out_type0*
_class
loc:@add_2/add
s
 gradients/add_2/add_grad/Shape_1Shapeactivation_2/Relu*
T0*
out_type0*
_class
loc:@add_2/add
°
.gradients/add_2/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2/add_grad/Shape gradients/add_2/add_grad/Shape_1*
T0*
_class
loc:@add_2/add
Ā
gradients/add_2/add_grad/SumSum)gradients/activation_4/Relu_grad/ReluGrad.gradients/add_2/add_grad/BroadcastGradientArgs*
T0*
_class
loc:@add_2/add*

Tidx0*
	keep_dims( 

 gradients/add_2/add_grad/ReshapeReshapegradients/add_2/add_grad/Sumgradients/add_2/add_grad/Shape*
T0*
Tshape0*
_class
loc:@add_2/add
Ę
gradients/add_2/add_grad/Sum_1Sum)gradients/activation_4/Relu_grad/ReluGrad0gradients/add_2/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_class
loc:@add_2/add
¤
"gradients/add_2/add_grad/Reshape_1Reshapegradients/add_2/add_grad/Sum_1 gradients/add_2/add_grad/Shape_1*
T0*
Tshape0*
_class
loc:@add_2/add
­
1gradients/res1b_branch2b/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/add_2/add_grad/Reshape*
data_formatNHWC*
T0*)
_class
loc:@res1b_branch2b/BiasAdd
Ģ
+gradients/res1b_branch2b/MatMul_grad/MatMulMatMul gradients/add_2/add_grad/Reshaperes1b_branch2b/kernel/read*
T0*(
_class
loc:@res1b_branch2b/MatMul*
transpose_a( *
transpose_b(
Å
-gradients/res1b_branch2b/MatMul_grad/MatMul_1MatMulactivation_3/Relu gradients/add_2/add_grad/Reshape*
transpose_a(*
transpose_b( *
T0*(
_class
loc:@res1b_branch2b/MatMul
¤
)gradients/activation_3/Relu_grad/ReluGradReluGrad+gradients/res1b_branch2b/MatMul_grad/MatMulactivation_3/Relu*
T0*$
_class
loc:@activation_3/Relu
¶
1gradients/res1b_branch2a/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/activation_3/Relu_grad/ReluGrad*
data_formatNHWC*
T0*)
_class
loc:@res1b_branch2a/BiasAdd
Õ
+gradients/res1b_branch2a/MatMul_grad/MatMulMatMul)gradients/activation_3/Relu_grad/ReluGradres1b_branch2a/kernel/read*
T0*(
_class
loc:@res1b_branch2a/MatMul*
transpose_a( *
transpose_b(
Ī
-gradients/res1b_branch2a/MatMul_grad/MatMul_1MatMulactivation_2/Relu)gradients/activation_3/Relu_grad/ReluGrad*
T0*(
_class
loc:@res1b_branch2a/MatMul*
transpose_a(*
transpose_b( 

gradients/AddNAddN"gradients/add_2/add_grad/Reshape_1+gradients/res1b_branch2a/MatMul_grad/MatMul*
_class
loc:@add_2/add*
N*
T0

)gradients/activation_2/Relu_grad/ReluGradReluGradgradients/AddNactivation_2/Relu*
T0*$
_class
loc:@activation_2/Relu
v
gradients/add_1/add_grad/ShapeShaperes1a_branch2b/BiasAdd*
T0*
out_type0*
_class
loc:@add_1/add
n
 gradients/add_1/add_grad/Shape_1Shapedense_1/Relu*
out_type0*
_class
loc:@add_1/add*
T0
°
.gradients/add_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1/add_grad/Shape gradients/add_1/add_grad/Shape_1*
T0*
_class
loc:@add_1/add
Ā
gradients/add_1/add_grad/SumSum)gradients/activation_2/Relu_grad/ReluGrad.gradients/add_1/add_grad/BroadcastGradientArgs*
_class
loc:@add_1/add*

Tidx0*
	keep_dims( *
T0

 gradients/add_1/add_grad/ReshapeReshapegradients/add_1/add_grad/Sumgradients/add_1/add_grad/Shape*
T0*
Tshape0*
_class
loc:@add_1/add
Ę
gradients/add_1/add_grad/Sum_1Sum)gradients/activation_2/Relu_grad/ReluGrad0gradients/add_1/add_grad/BroadcastGradientArgs:1*
T0*
_class
loc:@add_1/add*

Tidx0*
	keep_dims( 
¤
"gradients/add_1/add_grad/Reshape_1Reshapegradients/add_1/add_grad/Sum_1 gradients/add_1/add_grad/Shape_1*
T0*
Tshape0*
_class
loc:@add_1/add
­
1gradients/res1a_branch2b/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/add_1/add_grad/Reshape*
T0*)
_class
loc:@res1a_branch2b/BiasAdd*
data_formatNHWC
Ģ
+gradients/res1a_branch2b/MatMul_grad/MatMulMatMul gradients/add_1/add_grad/Reshaperes1a_branch2b/kernel/read*
transpose_a( *
transpose_b(*
T0*(
_class
loc:@res1a_branch2b/MatMul
Å
-gradients/res1a_branch2b/MatMul_grad/MatMul_1MatMulactivation_1/Relu gradients/add_1/add_grad/Reshape*
T0*(
_class
loc:@res1a_branch2b/MatMul*
transpose_a(*
transpose_b( 
¤
)gradients/activation_1/Relu_grad/ReluGradReluGrad+gradients/res1a_branch2b/MatMul_grad/MatMulactivation_1/Relu*
T0*$
_class
loc:@activation_1/Relu
¶
1gradients/res1a_branch2a/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/activation_1/Relu_grad/ReluGrad*)
_class
loc:@res1a_branch2a/BiasAdd*
data_formatNHWC*
T0
Õ
+gradients/res1a_branch2a/MatMul_grad/MatMulMatMul)gradients/activation_1/Relu_grad/ReluGradres1a_branch2a/kernel/read*
T0*(
_class
loc:@res1a_branch2a/MatMul*
transpose_a( *
transpose_b(
É
-gradients/res1a_branch2a/MatMul_grad/MatMul_1MatMuldense_1/Relu)gradients/activation_1/Relu_grad/ReluGrad*(
_class
loc:@res1a_branch2a/MatMul*
transpose_a(*
transpose_b( *
T0

gradients/AddN_1AddN"gradients/add_1/add_grad/Reshape_1+gradients/res1a_branch2a/MatMul_grad/MatMul*
_class
loc:@add_1/add*
N*
T0
z
$gradients/dense_1/Relu_grad/ReluGradReluGradgradients/AddN_1dense_1/Relu*
T0*
_class
loc:@dense_1/Relu
£
*gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*"
_class
loc:@dense_1/BiasAdd
»
$gradients/dense_1/MatMul_grad/MatMulMatMul$gradients/dense_1/Relu_grad/ReluGraddense_1/kernel/read*
T0*!
_class
loc:@dense_1/MatMul*
transpose_a( *
transpose_b(
±
&gradients/dense_1/MatMul_grad/MatMul_1MatMulinput_1$gradients/dense_1/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_1/MatMul*
transpose_a(*
transpose_b( 
<
AssignAdd/valueConst*
valueB
 *  ?*
dtype0
n
	AssignAdd	AssignAdd
iterationsAssignAdd/value*
use_locking( *
T0*
_class
loc:@iterations
2
add/yConst*
valueB
 *  ?*
dtype0
+
addAdditerations/readadd/y*
T0
%
PowPowbeta_2/readadd*
T0
4
sub_1/xConst*
valueB
 *  ?*
dtype0
#
sub_1Subsub_1/xPow*
T0
4
Const_3Const*
valueB
 *    *
dtype0
4
Const_4Const*
valueB
 *  *
dtype0
9
clip_by_value/MinimumMinimumsub_1Const_4*
T0
A
clip_by_valueMaximumclip_by_value/MinimumConst_3*
T0
$
SqrtSqrtclip_by_value*
T0
'
Pow_1Powbeta_1/readadd*
T0
4
sub_2/xConst*
valueB
 *  ?*
dtype0
%
sub_2Subsub_2/xPow_1*
T0
*
	truediv_1RealDivSqrtsub_2*
T0
)
mul_2Mullr/read	truediv_1*
T0
8
Const_5Const*
valueBd*    *
dtype0
X
Variable
VariableV2*
shape:d*
shared_name *
dtype0*
	container 
{
Variable/AssignAssignVariableConst_5*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
I
Variable/readIdentityVariable*
T0*
_class
loc:@Variable
<
Const_6Const*
dtype0*
valueBd*    
^

Variable_1
VariableV2*
shared_name *
dtype0*
	container *
shape
:d

Variable_1/AssignAssign
Variable_1Const_6*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(
O
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1
8
Const_7Const*
dtype0*
valueB*    
Z

Variable_2
VariableV2*
	container *
shape:*
shared_name *
dtype0

Variable_2/AssignAssign
Variable_2Const_7*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(
O
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2
<
Const_8Const*
valueBd*    *
dtype0
^

Variable_3
VariableV2*
shared_name *
dtype0*
	container *
shape
:d

Variable_3/AssignAssign
Variable_3Const_8*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(
O
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3
8
Const_9Const*
valueBd*    *
dtype0
Z

Variable_4
VariableV2*
dtype0*
	container *
shape:d*
shared_name 

Variable_4/AssignAssign
Variable_4Const_9*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_4
O
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4
=
Const_10Const*
valueBdd*    *
dtype0
^

Variable_5
VariableV2*
shape
:dd*
shared_name *
dtype0*
	container 

Variable_5/AssignAssign
Variable_5Const_10*
T0*
_class
loc:@Variable_5*
validate_shape(*
use_locking(
O
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0
9
Const_11Const*
valueBd*    *
dtype0
Z

Variable_6
VariableV2*
shared_name *
dtype0*
	container *
shape:d

Variable_6/AssignAssign
Variable_6Const_11*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(
O
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6
=
Const_12Const*
valueBdd*    *
dtype0
^

Variable_7
VariableV2*
shared_name *
dtype0*
	container *
shape
:dd

Variable_7/AssignAssign
Variable_7Const_12*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(
O
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7
9
Const_13Const*
valueBd*    *
dtype0
Z

Variable_8
VariableV2*
shared_name *
dtype0*
	container *
shape:d

Variable_8/AssignAssign
Variable_8Const_13*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(
O
Variable_8/readIdentity
Variable_8*
T0*
_class
loc:@Variable_8
=
Const_14Const*
valueBdd*    *
dtype0
^

Variable_9
VariableV2*
	container *
shape
:dd*
shared_name *
dtype0

Variable_9/AssignAssign
Variable_9Const_14*
T0*
_class
loc:@Variable_9*
validate_shape(*
use_locking(
O
Variable_9/readIdentity
Variable_9*
T0*
_class
loc:@Variable_9
9
Const_15Const*
valueBd*    *
dtype0
[
Variable_10
VariableV2*
	container *
shape:d*
shared_name *
dtype0

Variable_10/AssignAssignVariable_10Const_15*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(
R
Variable_10/readIdentityVariable_10*
T0*
_class
loc:@Variable_10
=
Const_16Const*
valueBdd*    *
dtype0
_
Variable_11
VariableV2*
shape
:dd*
shared_name *
dtype0*
	container 

Variable_11/AssignAssignVariable_11Const_16*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(
R
Variable_11/readIdentityVariable_11*
T0*
_class
loc:@Variable_11
9
Const_17Const*
valueBd*    *
dtype0
[
Variable_12
VariableV2*
dtype0*
	container *
shape:d*
shared_name 

Variable_12/AssignAssignVariable_12Const_17*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(
R
Variable_12/readIdentityVariable_12*
T0*
_class
loc:@Variable_12
=
Const_18Const*
valueBd*    *
dtype0
_
Variable_13
VariableV2*
dtype0*
	container *
shape
:d*
shared_name 

Variable_13/AssignAssignVariable_13Const_18*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(
R
Variable_13/readIdentityVariable_13*
T0*
_class
loc:@Variable_13
9
Const_19Const*
valueB*    *
dtype0
[
Variable_14
VariableV2*
dtype0*
	container *
shape:*
shared_name 

Variable_14/AssignAssignVariable_14Const_19*
_class
loc:@Variable_14*
validate_shape(*
use_locking(*
T0
R
Variable_14/readIdentityVariable_14*
T0*
_class
loc:@Variable_14
=
Const_20Const*
valueBd*    *
dtype0
_
Variable_15
VariableV2*
shared_name *
dtype0*
	container *
shape
:d

Variable_15/AssignAssignVariable_15Const_20*
_class
loc:@Variable_15*
validate_shape(*
use_locking(*
T0
R
Variable_15/readIdentityVariable_15*
_class
loc:@Variable_15*
T0
9
Const_21Const*
valueBd*    *
dtype0
[
Variable_16
VariableV2*
shape:d*
shared_name *
dtype0*
	container 

Variable_16/AssignAssignVariable_16Const_21*
T0*
_class
loc:@Variable_16*
validate_shape(*
use_locking(
R
Variable_16/readIdentityVariable_16*
T0*
_class
loc:@Variable_16
=
Const_22Const*
valueBdd*    *
dtype0
_
Variable_17
VariableV2*
dtype0*
	container *
shape
:dd*
shared_name 

Variable_17/AssignAssignVariable_17Const_22*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(
R
Variable_17/readIdentityVariable_17*
T0*
_class
loc:@Variable_17
9
Const_23Const*
valueBd*    *
dtype0
[
Variable_18
VariableV2*
dtype0*
	container *
shape:d*
shared_name 

Variable_18/AssignAssignVariable_18Const_23*
T0*
_class
loc:@Variable_18*
validate_shape(*
use_locking(
R
Variable_18/readIdentityVariable_18*
T0*
_class
loc:@Variable_18
=
Const_24Const*
valueBdd*    *
dtype0
_
Variable_19
VariableV2*
shape
:dd*
shared_name *
dtype0*
	container 

Variable_19/AssignAssignVariable_19Const_24*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(
R
Variable_19/readIdentityVariable_19*
T0*
_class
loc:@Variable_19
9
Const_25Const*
valueBd*    *
dtype0
[
Variable_20
VariableV2*
shared_name *
dtype0*
	container *
shape:d

Variable_20/AssignAssignVariable_20Const_25*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(
R
Variable_20/readIdentityVariable_20*
_class
loc:@Variable_20*
T0
=
Const_26Const*
valueBdd*    *
dtype0
_
Variable_21
VariableV2*
	container *
shape
:dd*
shared_name *
dtype0

Variable_21/AssignAssignVariable_21Const_26*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(
R
Variable_21/readIdentityVariable_21*
T0*
_class
loc:@Variable_21
9
Const_27Const*
valueBd*    *
dtype0
[
Variable_22
VariableV2*
shape:d*
shared_name *
dtype0*
	container 

Variable_22/AssignAssignVariable_22Const_27*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(
R
Variable_22/readIdentityVariable_22*
_class
loc:@Variable_22*
T0
=
Const_28Const*
valueBdd*    *
dtype0
_
Variable_23
VariableV2*
shared_name *
dtype0*
	container *
shape
:dd

Variable_23/AssignAssignVariable_23Const_28*
use_locking(*
T0*
_class
loc:@Variable_23*
validate_shape(
R
Variable_23/readIdentityVariable_23*
T0*
_class
loc:@Variable_23
1
mul_3Mulbeta_1/readVariable/read*
T0
4
sub_3/xConst*
valueB
 *  ?*
dtype0
+
sub_3Subsub_3/xbeta_1/read*
T0
H
mul_4Mulsub_3*gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0
#
add_3Addmul_3mul_4*
T0
4
mul_5Mulbeta_2/readVariable_12/read*
T0
4
sub_4/xConst*
valueB
 *  ?*
dtype0
+
sub_4Subsub_4/xbeta_2/read*
T0
G
Square_1Square*gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0
&
mul_6Mulsub_4Square_1*
T0
#
add_4Addmul_5mul_6*
T0
#
mul_7Mulmul_2add_3*
T0
5
Const_29Const*
valueB
 *    *
dtype0
5
Const_30Const*
valueB
 *  *
dtype0
<
clip_by_value_1/MinimumMinimumadd_4Const_30*
T0
F
clip_by_value_1Maximumclip_by_value_1/MinimumConst_29*
T0
(
Sqrt_1Sqrtclip_by_value_1*
T0
4
add_5/yConst*
valueB
 *wĢ+2*
dtype0
&
add_5AddSqrt_1add_5/y*
T0
+
	truediv_2RealDivmul_7add_5*
T0
3
sub_5Subdense_1/bias/read	truediv_2*
T0
p
AssignAssignVariableadd_3*
_class
loc:@Variable*
validate_shape(*
use_locking(*
T0
x
Assign_1AssignVariable_12add_4*
_class
loc:@Variable_12*
validate_shape(*
use_locking(*
T0
z
Assign_2Assigndense_1/biassub_5*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(
3
mul_8Mulbeta_1/readVariable_1/read*
T0
4
sub_6/xConst*
valueB
 *  ?*
dtype0
+
sub_6Subsub_6/xbeta_1/read*
T0
D
mul_9Mulsub_6&gradients/dense_1/MatMul_grad/MatMul_1*
T0
#
add_6Addmul_8mul_9*
T0
5
mul_10Mulbeta_2/readVariable_13/read*
T0
4
sub_7/xConst*
valueB
 *  ?*
dtype0
+
sub_7Subsub_7/xbeta_2/read*
T0
C
Square_2Square&gradients/dense_1/MatMul_grad/MatMul_1*
T0
'
mul_11Mulsub_7Square_2*
T0
%
add_7Addmul_10mul_11*
T0
$
mul_12Mulmul_2add_6*
T0
5
Const_31Const*
valueB
 *    *
dtype0
5
Const_32Const*
valueB
 *  *
dtype0
<
clip_by_value_2/MinimumMinimumadd_7Const_32*
T0
F
clip_by_value_2Maximumclip_by_value_2/MinimumConst_31*
T0
(
Sqrt_2Sqrtclip_by_value_2*
T0
4
add_8/yConst*
valueB
 *wĢ+2*
dtype0
&
add_8AddSqrt_2add_8/y*
T0
,
	truediv_3RealDivmul_12add_8*
T0
5
sub_8Subdense_1/kernel/read	truediv_3*
T0
v
Assign_3Assign
Variable_1add_6*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(
x
Assign_4AssignVariable_13add_7*
T0*
_class
loc:@Variable_13*
validate_shape(*
use_locking(
~
Assign_5Assigndense_1/kernelsub_8*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(
4
mul_13Mulbeta_1/readVariable_2/read*
T0
4
sub_9/xConst*
valueB
 *  ?*
dtype0
+
sub_9Subsub_9/xbeta_1/read*
T0
I
mul_14Mulsub_9*gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0
%
add_9Addmul_13mul_14*
T0
5
mul_15Mulbeta_2/readVariable_14/read*
T0
5
sub_10/xConst*
valueB
 *  ?*
dtype0
-
sub_10Subsub_10/xbeta_2/read*
T0
G
Square_3Square*gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0
(
mul_16Mulsub_10Square_3*
T0
&
add_10Addmul_15mul_16*
T0
$
mul_17Mulmul_2add_9*
T0
5
Const_33Const*
valueB
 *    *
dtype0
5
Const_34Const*
valueB
 *  *
dtype0
=
clip_by_value_3/MinimumMinimumadd_10Const_34*
T0
F
clip_by_value_3Maximumclip_by_value_3/MinimumConst_33*
T0
(
Sqrt_3Sqrtclip_by_value_3*
T0
5
add_11/yConst*
valueB
 *wĢ+2*
dtype0
(
add_11AddSqrt_3add_11/y*
T0
-
	truediv_4RealDivmul_17add_11*
T0
4
sub_11Subdense_2/bias/read	truediv_4*
T0
v
Assign_6Assign
Variable_2add_9*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(
y
Assign_7AssignVariable_14add_10*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(
{
Assign_8Assigndense_2/biassub_11*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(
4
mul_18Mulbeta_1/readVariable_3/read*
T0
5
sub_12/xConst*
valueB
 *  ?*
dtype0
-
sub_12Subsub_12/xbeta_1/read*
T0
F
mul_19Mulsub_12&gradients/dense_2/MatMul_grad/MatMul_1*
T0
&
add_12Addmul_18mul_19*
T0
5
mul_20Mulbeta_2/readVariable_15/read*
T0
5
sub_13/xConst*
dtype0*
valueB
 *  ?
-
sub_13Subsub_13/xbeta_2/read*
T0
C
Square_4Square&gradients/dense_2/MatMul_grad/MatMul_1*
T0
(
mul_21Mulsub_13Square_4*
T0
&
add_13Addmul_20mul_21*
T0
%
mul_22Mulmul_2add_12*
T0
5
Const_35Const*
valueB
 *    *
dtype0
5
Const_36Const*
valueB
 *  *
dtype0
=
clip_by_value_4/MinimumMinimumadd_13Const_36*
T0
F
clip_by_value_4Maximumclip_by_value_4/MinimumConst_35*
T0
(
Sqrt_4Sqrtclip_by_value_4*
T0
5
add_14/yConst*
valueB
 *wĢ+2*
dtype0
(
add_14AddSqrt_4add_14/y*
T0
-
	truediv_5RealDivmul_22add_14*
T0
6
sub_14Subdense_2/kernel/read	truediv_5*
T0
w
Assign_9Assign
Variable_3add_12*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(
z
	Assign_10AssignVariable_15add_13*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(

	Assign_11Assigndense_2/kernelsub_14*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(
4
mul_23Mulbeta_1/readVariable_4/read*
T0
5
sub_15/xConst*
valueB
 *  ?*
dtype0
-
sub_15Subsub_15/xbeta_1/read*
T0
Q
mul_24Mulsub_151gradients/res1a_branch2a/BiasAdd_grad/BiasAddGrad*
T0
&
add_15Addmul_23mul_24*
T0
5
mul_25Mulbeta_2/readVariable_16/read*
T0
5
sub_16/xConst*
valueB
 *  ?*
dtype0
-
sub_16Subsub_16/xbeta_2/read*
T0
N
Square_5Square1gradients/res1a_branch2a/BiasAdd_grad/BiasAddGrad*
T0
(
mul_26Mulsub_16Square_5*
T0
&
add_16Addmul_25mul_26*
T0
%
mul_27Mulmul_2add_15*
T0
5
Const_37Const*
dtype0*
valueB
 *    
5
Const_38Const*
dtype0*
valueB
 *  
=
clip_by_value_5/MinimumMinimumadd_16Const_38*
T0
F
clip_by_value_5Maximumclip_by_value_5/MinimumConst_37*
T0
(
Sqrt_5Sqrtclip_by_value_5*
T0
5
add_17/yConst*
valueB
 *wĢ+2*
dtype0
(
add_17AddSqrt_5add_17/y*
T0
-
	truediv_6RealDivmul_27add_17*
T0
;
sub_17Subres1a_branch2a/bias/read	truediv_6*
T0
x
	Assign_12Assign
Variable_4add_15*
T0*
_class
loc:@Variable_4*
validate_shape(*
use_locking(
z
	Assign_13AssignVariable_16add_16*
T0*
_class
loc:@Variable_16*
validate_shape(*
use_locking(

	Assign_14Assignres1a_branch2a/biassub_17*
use_locking(*
T0*&
_class
loc:@res1a_branch2a/bias*
validate_shape(
4
mul_28Mulbeta_1/readVariable_5/read*
T0
5
sub_18/xConst*
valueB
 *  ?*
dtype0
-
sub_18Subsub_18/xbeta_1/read*
T0
M
mul_29Mulsub_18-gradients/res1a_branch2a/MatMul_grad/MatMul_1*
T0
&
add_18Addmul_28mul_29*
T0
5
mul_30Mulbeta_2/readVariable_17/read*
T0
5
sub_19/xConst*
valueB
 *  ?*
dtype0
-
sub_19Subsub_19/xbeta_2/read*
T0
J
Square_6Square-gradients/res1a_branch2a/MatMul_grad/MatMul_1*
T0
(
mul_31Mulsub_19Square_6*
T0
&
add_19Addmul_30mul_31*
T0
%
mul_32Mulmul_2add_18*
T0
5
Const_39Const*
dtype0*
valueB
 *    
5
Const_40Const*
valueB
 *  *
dtype0
=
clip_by_value_6/MinimumMinimumadd_19Const_40*
T0
F
clip_by_value_6Maximumclip_by_value_6/MinimumConst_39*
T0
(
Sqrt_6Sqrtclip_by_value_6*
T0
5
add_20/yConst*
valueB
 *wĢ+2*
dtype0
(
add_20AddSqrt_6add_20/y*
T0
-
	truediv_7RealDivmul_32add_20*
T0
=
sub_20Subres1a_branch2a/kernel/read	truediv_7*
T0
x
	Assign_15Assign
Variable_5add_18*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(
z
	Assign_16AssignVariable_17add_19*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(

	Assign_17Assignres1a_branch2a/kernelsub_20*
use_locking(*
T0*(
_class
loc:@res1a_branch2a/kernel*
validate_shape(
4
mul_33Mulbeta_1/readVariable_6/read*
T0
5
sub_21/xConst*
valueB
 *  ?*
dtype0
-
sub_21Subsub_21/xbeta_1/read*
T0
Q
mul_34Mulsub_211gradients/res1a_branch2b/BiasAdd_grad/BiasAddGrad*
T0
&
add_21Addmul_33mul_34*
T0
5
mul_35Mulbeta_2/readVariable_18/read*
T0
5
sub_22/xConst*
valueB
 *  ?*
dtype0
-
sub_22Subsub_22/xbeta_2/read*
T0
N
Square_7Square1gradients/res1a_branch2b/BiasAdd_grad/BiasAddGrad*
T0
(
mul_36Mulsub_22Square_7*
T0
&
add_22Addmul_35mul_36*
T0
%
mul_37Mulmul_2add_21*
T0
5
Const_41Const*
valueB
 *    *
dtype0
5
Const_42Const*
valueB
 *  *
dtype0
=
clip_by_value_7/MinimumMinimumadd_22Const_42*
T0
F
clip_by_value_7Maximumclip_by_value_7/MinimumConst_41*
T0
(
Sqrt_7Sqrtclip_by_value_7*
T0
5
add_23/yConst*
valueB
 *wĢ+2*
dtype0
(
add_23AddSqrt_7add_23/y*
T0
-
	truediv_8RealDivmul_37add_23*
T0
;
sub_23Subres1a_branch2b/bias/read	truediv_8*
T0
x
	Assign_18Assign
Variable_6add_21*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(
z
	Assign_19AssignVariable_18add_22*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_18

	Assign_20Assignres1a_branch2b/biassub_23*&
_class
loc:@res1a_branch2b/bias*
validate_shape(*
use_locking(*
T0
4
mul_38Mulbeta_1/readVariable_7/read*
T0
5
sub_24/xConst*
valueB
 *  ?*
dtype0
-
sub_24Subsub_24/xbeta_1/read*
T0
M
mul_39Mulsub_24-gradients/res1a_branch2b/MatMul_grad/MatMul_1*
T0
&
add_24Addmul_38mul_39*
T0
5
mul_40Mulbeta_2/readVariable_19/read*
T0
5
sub_25/xConst*
valueB
 *  ?*
dtype0
-
sub_25Subsub_25/xbeta_2/read*
T0
J
Square_8Square-gradients/res1a_branch2b/MatMul_grad/MatMul_1*
T0
(
mul_41Mulsub_25Square_8*
T0
&
add_25Addmul_40mul_41*
T0
%
mul_42Mulmul_2add_24*
T0
5
Const_43Const*
valueB
 *    *
dtype0
5
Const_44Const*
valueB
 *  *
dtype0
=
clip_by_value_8/MinimumMinimumadd_25Const_44*
T0
F
clip_by_value_8Maximumclip_by_value_8/MinimumConst_43*
T0
(
Sqrt_8Sqrtclip_by_value_8*
T0
5
add_26/yConst*
dtype0*
valueB
 *wĢ+2
(
add_26AddSqrt_8add_26/y*
T0
-
	truediv_9RealDivmul_42add_26*
T0
=
sub_26Subres1a_branch2b/kernel/read	truediv_9*
T0
x
	Assign_21Assign
Variable_7add_24*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(
z
	Assign_22AssignVariable_19add_25*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(

	Assign_23Assignres1a_branch2b/kernelsub_26*
use_locking(*
T0*(
_class
loc:@res1a_branch2b/kernel*
validate_shape(
4
mul_43Mulbeta_1/readVariable_8/read*
T0
5
sub_27/xConst*
valueB
 *  ?*
dtype0
-
sub_27Subsub_27/xbeta_1/read*
T0
Q
mul_44Mulsub_271gradients/res1b_branch2a/BiasAdd_grad/BiasAddGrad*
T0
&
add_27Addmul_43mul_44*
T0
5
mul_45Mulbeta_2/readVariable_20/read*
T0
5
sub_28/xConst*
valueB
 *  ?*
dtype0
-
sub_28Subsub_28/xbeta_2/read*
T0
N
Square_9Square1gradients/res1b_branch2a/BiasAdd_grad/BiasAddGrad*
T0
(
mul_46Mulsub_28Square_9*
T0
&
add_28Addmul_45mul_46*
T0
%
mul_47Mulmul_2add_27*
T0
5
Const_45Const*
valueB
 *    *
dtype0
5
Const_46Const*
valueB
 *  *
dtype0
=
clip_by_value_9/MinimumMinimumadd_28Const_46*
T0
F
clip_by_value_9Maximumclip_by_value_9/MinimumConst_45*
T0
(
Sqrt_9Sqrtclip_by_value_9*
T0
5
add_29/yConst*
valueB
 *wĢ+2*
dtype0
(
add_29AddSqrt_9add_29/y*
T0
.

truediv_10RealDivmul_47add_29*
T0
<
sub_29Subres1b_branch2a/bias/read
truediv_10*
T0
x
	Assign_24Assign
Variable_8add_27*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(
z
	Assign_25AssignVariable_20add_28*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(

	Assign_26Assignres1b_branch2a/biassub_29*
use_locking(*
T0*&
_class
loc:@res1b_branch2a/bias*
validate_shape(
4
mul_48Mulbeta_1/readVariable_9/read*
T0
5
sub_30/xConst*
valueB
 *  ?*
dtype0
-
sub_30Subsub_30/xbeta_1/read*
T0
M
mul_49Mulsub_30-gradients/res1b_branch2a/MatMul_grad/MatMul_1*
T0
&
add_30Addmul_48mul_49*
T0
5
mul_50Mulbeta_2/readVariable_21/read*
T0
5
sub_31/xConst*
valueB
 *  ?*
dtype0
-
sub_31Subsub_31/xbeta_2/read*
T0
K
	Square_10Square-gradients/res1b_branch2a/MatMul_grad/MatMul_1*
T0
)
mul_51Mulsub_31	Square_10*
T0
&
add_31Addmul_50mul_51*
T0
%
mul_52Mulmul_2add_30*
T0
5
Const_47Const*
valueB
 *    *
dtype0
5
Const_48Const*
valueB
 *  *
dtype0
>
clip_by_value_10/MinimumMinimumadd_31Const_48*
T0
H
clip_by_value_10Maximumclip_by_value_10/MinimumConst_47*
T0
*
Sqrt_10Sqrtclip_by_value_10*
T0
5
add_32/yConst*
valueB
 *wĢ+2*
dtype0
)
add_32AddSqrt_10add_32/y*
T0
.

truediv_11RealDivmul_52add_32*
T0
>
sub_32Subres1b_branch2a/kernel/read
truediv_11*
T0
x
	Assign_27Assign
Variable_9add_30*
T0*
_class
loc:@Variable_9*
validate_shape(*
use_locking(
z
	Assign_28AssignVariable_21add_31*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(

	Assign_29Assignres1b_branch2a/kernelsub_32*
T0*(
_class
loc:@res1b_branch2a/kernel*
validate_shape(*
use_locking(
5
mul_53Mulbeta_1/readVariable_10/read*
T0
5
sub_33/xConst*
valueB
 *  ?*
dtype0
-
sub_33Subsub_33/xbeta_1/read*
T0
Q
mul_54Mulsub_331gradients/res1b_branch2b/BiasAdd_grad/BiasAddGrad*
T0
&
add_33Addmul_53mul_54*
T0
5
mul_55Mulbeta_2/readVariable_22/read*
T0
5
sub_34/xConst*
dtype0*
valueB
 *  ?
-
sub_34Subsub_34/xbeta_2/read*
T0
O
	Square_11Square1gradients/res1b_branch2b/BiasAdd_grad/BiasAddGrad*
T0
)
mul_56Mulsub_34	Square_11*
T0
&
add_34Addmul_55mul_56*
T0
%
mul_57Mulmul_2add_33*
T0
5
Const_49Const*
valueB
 *    *
dtype0
5
Const_50Const*
valueB
 *  *
dtype0
>
clip_by_value_11/MinimumMinimumadd_34Const_50*
T0
H
clip_by_value_11Maximumclip_by_value_11/MinimumConst_49*
T0
*
Sqrt_11Sqrtclip_by_value_11*
T0
5
add_35/yConst*
valueB
 *wĢ+2*
dtype0
)
add_35AddSqrt_11add_35/y*
T0
.

truediv_12RealDivmul_57add_35*
T0
<
sub_35Subres1b_branch2b/bias/read
truediv_12*
T0
z
	Assign_30AssignVariable_10add_33*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_10
z
	Assign_31AssignVariable_22add_34*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(

	Assign_32Assignres1b_branch2b/biassub_35*&
_class
loc:@res1b_branch2b/bias*
validate_shape(*
use_locking(*
T0
5
mul_58Mulbeta_1/readVariable_11/read*
T0
5
sub_36/xConst*
valueB
 *  ?*
dtype0
-
sub_36Subsub_36/xbeta_1/read*
T0
M
mul_59Mulsub_36-gradients/res1b_branch2b/MatMul_grad/MatMul_1*
T0
&
add_36Addmul_58mul_59*
T0
5
mul_60Mulbeta_2/readVariable_23/read*
T0
5
sub_37/xConst*
valueB
 *  ?*
dtype0
-
sub_37Subsub_37/xbeta_2/read*
T0
K
	Square_12Square-gradients/res1b_branch2b/MatMul_grad/MatMul_1*
T0
)
mul_61Mulsub_37	Square_12*
T0
&
add_37Addmul_60mul_61*
T0
%
mul_62Mulmul_2add_36*
T0
5
Const_51Const*
valueB
 *    *
dtype0
5
Const_52Const*
valueB
 *  *
dtype0
>
clip_by_value_12/MinimumMinimumadd_37Const_52*
T0
H
clip_by_value_12Maximumclip_by_value_12/MinimumConst_51*
T0
*
Sqrt_12Sqrtclip_by_value_12*
T0
5
add_38/yConst*
valueB
 *wĢ+2*
dtype0
)
add_38AddSqrt_12add_38/y*
T0
.

truediv_13RealDivmul_62add_38*
T0
>
sub_38Subres1b_branch2b/kernel/read
truediv_13*
T0
z
	Assign_33AssignVariable_11add_36*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(
z
	Assign_34AssignVariable_23add_37*
_class
loc:@Variable_23*
validate_shape(*
use_locking(*
T0

	Assign_35Assignres1b_branch2b/kernelsub_38*
T0*(
_class
loc:@res1b_branch2b/kernel*
validate_shape(*
use_locking(
Õ
group_deps_1NoOp^Assign
^AssignAdd	^Assign_1
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19	^Assign_2
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29	^Assign_3
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9^Mean_4^mul_1

initNoOp^Variable/Assign^Variable_1/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_16/Assign^Variable_17/Assign^Variable_18/Assign^Variable_19/Assign^Variable_2/Assign^Variable_20/Assign^Variable_21/Assign^Variable_22/Assign^Variable_23/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^beta_1/Assign^beta_2/Assign^decay/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^iterations/Assign
^lr/Assign^res1a_branch2a/bias/Assign^res1a_branch2a/kernel/Assign^res1a_branch2b/bias/Assign^res1a_branch2b/kernel/Assign^res1b_branch2a/bias/Assign^res1b_branch2a/kernel/Assign^res1b_branch2b/bias/Assign^res1b_branch2b/kernel/Assign
<
PlaceholderPlaceholder*
dtype0*
shape
:d

	Assign_36Assigndense_1/kernelPlaceholder*!
_class
loc:@dense_1/kernel*
validate_shape(*
use_locking( *
T0
:
Placeholder_1Placeholder*
shape:d*
dtype0

	Assign_37Assigndense_1/biasPlaceholder_1*
use_locking( *
T0*
_class
loc:@dense_1/bias*
validate_shape(
>
Placeholder_2Placeholder*
dtype0*
shape
:dd

	Assign_38Assignres1a_branch2a/kernelPlaceholder_2*
T0*(
_class
loc:@res1a_branch2a/kernel*
validate_shape(*
use_locking( 
:
Placeholder_3Placeholder*
shape:d*
dtype0

	Assign_39Assignres1a_branch2a/biasPlaceholder_3*&
_class
loc:@res1a_branch2a/bias*
validate_shape(*
use_locking( *
T0
>
Placeholder_4Placeholder*
dtype0*
shape
:dd

	Assign_40Assignres1a_branch2b/kernelPlaceholder_4*
validate_shape(*
use_locking( *
T0*(
_class
loc:@res1a_branch2b/kernel
:
Placeholder_5Placeholder*
dtype0*
shape:d

	Assign_41Assignres1a_branch2b/biasPlaceholder_5*
use_locking( *
T0*&
_class
loc:@res1a_branch2b/bias*
validate_shape(
>
Placeholder_6Placeholder*
dtype0*
shape
:dd

	Assign_42Assignres1b_branch2a/kernelPlaceholder_6*
use_locking( *
T0*(
_class
loc:@res1b_branch2a/kernel*
validate_shape(
:
Placeholder_7Placeholder*
shape:d*
dtype0

	Assign_43Assignres1b_branch2a/biasPlaceholder_7*
use_locking( *
T0*&
_class
loc:@res1b_branch2a/bias*
validate_shape(
>
Placeholder_8Placeholder*
shape
:dd*
dtype0

	Assign_44Assignres1b_branch2b/kernelPlaceholder_8*
validate_shape(*
use_locking( *
T0*(
_class
loc:@res1b_branch2b/kernel
:
Placeholder_9Placeholder*
dtype0*
shape:d

	Assign_45Assignres1b_branch2b/biasPlaceholder_9*
T0*&
_class
loc:@res1b_branch2b/bias*
validate_shape(*
use_locking( 
?
Placeholder_10Placeholder*
dtype0*
shape
:d

	Assign_46Assigndense_2/kernelPlaceholder_10*
use_locking( *
T0*!
_class
loc:@dense_2/kernel*
validate_shape(
;
Placeholder_11Placeholder*
shape:*
dtype0

	Assign_47Assigndense_2/biasPlaceholder_11*
use_locking( *
T0*
_class
loc:@dense_2/bias*
validate_shape(
&
group_deps_2NoOp^dense_2/BiasAdd
8

save/ConstConst*
dtype0*
valueB Bmodel

save/SaveV2/tensor_namesConst*Ō
valueŹBĒ)BVariableB
Variable_1BVariable_10BVariable_11BVariable_12BVariable_13BVariable_14BVariable_15BVariable_16BVariable_17BVariable_18BVariable_19B
Variable_2BVariable_20BVariable_21BVariable_22BVariable_23B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7B
Variable_8B
Variable_9Bbeta_1Bbeta_2BdecayBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelB
iterationsBlrBres1a_branch2a/biasBres1a_branch2a/kernelBres1a_branch2b/biasBres1a_branch2b/kernelBres1b_branch2a/biasBres1b_branch2a/kernelBres1b_branch2b/biasBres1b_branch2b/kernel*
dtype0

save/SaveV2/shape_and_slicesConst*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ń
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable
Variable_1Variable_10Variable_11Variable_12Variable_13Variable_14Variable_15Variable_16Variable_17Variable_18Variable_19
Variable_2Variable_20Variable_21Variable_22Variable_23
Variable_3
Variable_4
Variable_5
Variable_6
Variable_7
Variable_8
Variable_9beta_1beta_2decaydense_1/biasdense_1/kerneldense_2/biasdense_2/kernel
iterationslrres1a_branch2a/biasres1a_branch2a/kernelres1a_branch2b/biasres1a_branch2b/kernelres1b_branch2a/biasres1b_branch2a/kernelres1b_branch2b/biasres1b_branch2b/kernel*7
dtypes-
+2)
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const

save/RestoreV2/tensor_namesConst"/device:CPU:0*Ō
valueŹBĒ)BVariableB
Variable_1BVariable_10BVariable_11BVariable_12BVariable_13BVariable_14BVariable_15BVariable_16BVariable_17BVariable_18BVariable_19B
Variable_2BVariable_20BVariable_21BVariable_22BVariable_23B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7B
Variable_8B
Variable_9Bbeta_1Bbeta_2BdecayBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelB
iterationsBlrBres1a_branch2a/biasBres1a_branch2a/kernelBres1a_branch2b/biasBres1a_branch2b/kernelBres1b_branch2a/biasBres1b_branch2a/kernelBres1b_branch2b/biasBres1b_branch2b/kernel*
dtype0
«
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
­
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*7
dtypes-
+2)
~
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(

save/Assign_1Assign
Variable_1save/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(

save/Assign_2AssignVariable_10save/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(

save/Assign_3AssignVariable_11save/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(

save/Assign_4AssignVariable_12save/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(

save/Assign_5AssignVariable_13save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(

save/Assign_6AssignVariable_14save/RestoreV2:6*
_class
loc:@Variable_14*
validate_shape(*
use_locking(*
T0

save/Assign_7AssignVariable_15save/RestoreV2:7*
_class
loc:@Variable_15*
validate_shape(*
use_locking(*
T0

save/Assign_8AssignVariable_16save/RestoreV2:8*
_class
loc:@Variable_16*
validate_shape(*
use_locking(*
T0

save/Assign_9AssignVariable_17save/RestoreV2:9*
_class
loc:@Variable_17*
validate_shape(*
use_locking(*
T0

save/Assign_10AssignVariable_18save/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(

save/Assign_11AssignVariable_19save/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(

save/Assign_12Assign
Variable_2save/RestoreV2:12*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(

save/Assign_13AssignVariable_20save/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(

save/Assign_14AssignVariable_21save/RestoreV2:14*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(

save/Assign_15AssignVariable_22save/RestoreV2:15*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(

save/Assign_16AssignVariable_23save/RestoreV2:16*
use_locking(*
T0*
_class
loc:@Variable_23*
validate_shape(

save/Assign_17Assign
Variable_3save/RestoreV2:17*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(

save/Assign_18Assign
Variable_4save/RestoreV2:18*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(

save/Assign_19Assign
Variable_5save/RestoreV2:19*
_class
loc:@Variable_5*
validate_shape(*
use_locking(*
T0

save/Assign_20Assign
Variable_6save/RestoreV2:20*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(

save/Assign_21Assign
Variable_7save/RestoreV2:21*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_7

save/Assign_22Assign
Variable_8save/RestoreV2:22*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(

save/Assign_23Assign
Variable_9save/RestoreV2:23*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(

save/Assign_24Assignbeta_1save/RestoreV2:24*
use_locking(*
T0*
_class
loc:@beta_1*
validate_shape(

save/Assign_25Assignbeta_2save/RestoreV2:25*
T0*
_class
loc:@beta_2*
validate_shape(*
use_locking(
~
save/Assign_26Assigndecaysave/RestoreV2:26*
use_locking(*
T0*
_class

loc:@decay*
validate_shape(

save/Assign_27Assigndense_1/biassave/RestoreV2:27*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
use_locking(

save/Assign_28Assigndense_1/kernelsave/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(

save/Assign_29Assigndense_2/biassave/RestoreV2:29*
_class
loc:@dense_2/bias*
validate_shape(*
use_locking(*
T0

save/Assign_30Assigndense_2/kernelsave/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(

save/Assign_31Assign
iterationssave/RestoreV2:31*
use_locking(*
T0*
_class
loc:@iterations*
validate_shape(
x
save/Assign_32Assignlrsave/RestoreV2:32*
use_locking(*
T0*
_class
	loc:@lr*
validate_shape(

save/Assign_33Assignres1a_branch2a/biassave/RestoreV2:33*
validate_shape(*
use_locking(*
T0*&
_class
loc:@res1a_branch2a/bias

save/Assign_34Assignres1a_branch2a/kernelsave/RestoreV2:34*
T0*(
_class
loc:@res1a_branch2a/kernel*
validate_shape(*
use_locking(

save/Assign_35Assignres1a_branch2b/biassave/RestoreV2:35*
validate_shape(*
use_locking(*
T0*&
_class
loc:@res1a_branch2b/bias

save/Assign_36Assignres1a_branch2b/kernelsave/RestoreV2:36*
use_locking(*
T0*(
_class
loc:@res1a_branch2b/kernel*
validate_shape(

save/Assign_37Assignres1b_branch2a/biassave/RestoreV2:37*
use_locking(*
T0*&
_class
loc:@res1b_branch2a/bias*
validate_shape(

save/Assign_38Assignres1b_branch2a/kernelsave/RestoreV2:38*(
_class
loc:@res1b_branch2a/kernel*
validate_shape(*
use_locking(*
T0

save/Assign_39Assignres1b_branch2b/biassave/RestoreV2:39*
use_locking(*
T0*&
_class
loc:@res1b_branch2b/bias*
validate_shape(

save/Assign_40Assignres1b_branch2b/kernelsave/RestoreV2:40*
T0*(
_class
loc:@res1b_branch2b/kernel*
validate_shape(*
use_locking(
Å
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
A
input_2Placeholder*
shape:’’’’’’’’’*
dtype0
Q
dense_3/random_uniform/shapeConst*
dtype0*
valueB"   X  
G
dense_3/random_uniform/minConst*
valueB
 *£uĢ½*
dtype0
G
dense_3/random_uniform/maxConst*
valueB
 *£uĢ=*
dtype0

$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0*
seed2Ė«ä*
seed±’å)*
T0
b
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0
l
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0
^
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0
c
dense_3/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape:	Ų

dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(
[
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel
?
dense_3/ConstConst*
valueBŲ*    *
dtype0
]
dense_3/bias
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 

dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
use_locking(
U
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias
e
dense_3/MatMulMatMulinput_2dense_3/kernel/read*
transpose_a( *
transpose_b( *
T0
]
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC
.
dense_3/ReluReludense_3/BiasAdd*
T0
Z
%res1a_branch2a_1/random_uniform/shapeConst*
valueB"X  X  *
dtype0
P
#res1a_branch2a_1/random_uniform/minConst*
valueB
 *ĆŠ½*
dtype0
P
#res1a_branch2a_1/random_uniform/maxConst*
dtype0*
valueB
 *ĆŠ=

-res1a_branch2a_1/random_uniform/RandomUniformRandomUniform%res1a_branch2a_1/random_uniform/shape*
seed2Ż°*
seed±’å)*
T0*
dtype0
}
#res1a_branch2a_1/random_uniform/subSub#res1a_branch2a_1/random_uniform/max#res1a_branch2a_1/random_uniform/min*
T0

#res1a_branch2a_1/random_uniform/mulMul-res1a_branch2a_1/random_uniform/RandomUniform#res1a_branch2a_1/random_uniform/sub*
T0
y
res1a_branch2a_1/random_uniformAdd#res1a_branch2a_1/random_uniform/mul#res1a_branch2a_1/random_uniform/min*
T0
m
res1a_branch2a_1/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape:
ŲŲ
Ą
res1a_branch2a_1/kernel/AssignAssignres1a_branch2a_1/kernelres1a_branch2a_1/random_uniform*
use_locking(*
T0**
_class 
loc:@res1a_branch2a_1/kernel*
validate_shape(
v
res1a_branch2a_1/kernel/readIdentityres1a_branch2a_1/kernel*
T0**
_class 
loc:@res1a_branch2a_1/kernel
H
res1a_branch2a_1/ConstConst*
dtype0*
valueBŲ*    
f
res1a_branch2a_1/bias
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 
±
res1a_branch2a_1/bias/AssignAssignres1a_branch2a_1/biasres1a_branch2a_1/Const*
use_locking(*
T0*(
_class
loc:@res1a_branch2a_1/bias*
validate_shape(
p
res1a_branch2a_1/bias/readIdentityres1a_branch2a_1/bias*(
_class
loc:@res1a_branch2a_1/bias*
T0
|
res1a_branch2a_1/MatMulMatMuldense_3/Relures1a_branch2a_1/kernel/read*
T0*
transpose_a( *
transpose_b( 
x
res1a_branch2a_1/BiasAddBiasAddres1a_branch2a_1/MatMulres1a_branch2a_1/bias/read*
T0*
data_formatNHWC
<
activation_5/ReluRelures1a_branch2a_1/BiasAdd*
T0
Z
%res1a_branch2b_1/random_uniform/shapeConst*
valueB"X  X  *
dtype0
P
#res1a_branch2b_1/random_uniform/minConst*
valueB
 *ĆŠ½*
dtype0
P
#res1a_branch2b_1/random_uniform/maxConst*
valueB
 *ĆŠ=*
dtype0

-res1a_branch2b_1/random_uniform/RandomUniformRandomUniform%res1a_branch2b_1/random_uniform/shape*
dtype0*
seed2¼Ć*
seed±’å)*
T0
}
#res1a_branch2b_1/random_uniform/subSub#res1a_branch2b_1/random_uniform/max#res1a_branch2b_1/random_uniform/min*
T0

#res1a_branch2b_1/random_uniform/mulMul-res1a_branch2b_1/random_uniform/RandomUniform#res1a_branch2b_1/random_uniform/sub*
T0
y
res1a_branch2b_1/random_uniformAdd#res1a_branch2b_1/random_uniform/mul#res1a_branch2b_1/random_uniform/min*
T0
m
res1a_branch2b_1/kernel
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 
Ą
res1a_branch2b_1/kernel/AssignAssignres1a_branch2b_1/kernelres1a_branch2b_1/random_uniform**
_class 
loc:@res1a_branch2b_1/kernel*
validate_shape(*
use_locking(*
T0
v
res1a_branch2b_1/kernel/readIdentityres1a_branch2b_1/kernel*
T0**
_class 
loc:@res1a_branch2b_1/kernel
H
res1a_branch2b_1/ConstConst*
valueBŲ*    *
dtype0
f
res1a_branch2b_1/bias
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 
±
res1a_branch2b_1/bias/AssignAssignres1a_branch2b_1/biasres1a_branch2b_1/Const*
use_locking(*
T0*(
_class
loc:@res1a_branch2b_1/bias*
validate_shape(
p
res1a_branch2b_1/bias/readIdentityres1a_branch2b_1/bias*
T0*(
_class
loc:@res1a_branch2b_1/bias

res1a_branch2b_1/MatMulMatMulactivation_5/Relures1a_branch2b_1/kernel/read*
transpose_b( *
T0*
transpose_a( 
x
res1a_branch2b_1/BiasAddBiasAddres1a_branch2b_1/MatMulres1a_branch2b_1/bias/read*
data_formatNHWC*
T0
C
add_3_1/addAddres1a_branch2b_1/BiasAdddense_3/Relu*
T0
/
activation_6/ReluReluadd_3_1/add*
T0
Z
%res1b_branch2a_1/random_uniform/shapeConst*
dtype0*
valueB"X  X  
P
#res1b_branch2a_1/random_uniform/minConst*
dtype0*
valueB
 *ĆŠ½
P
#res1b_branch2a_1/random_uniform/maxConst*
valueB
 *ĆŠ=*
dtype0

-res1b_branch2a_1/random_uniform/RandomUniformRandomUniform%res1b_branch2a_1/random_uniform/shape*
dtype0*
seed2Ō«*
seed±’å)*
T0
}
#res1b_branch2a_1/random_uniform/subSub#res1b_branch2a_1/random_uniform/max#res1b_branch2a_1/random_uniform/min*
T0

#res1b_branch2a_1/random_uniform/mulMul-res1b_branch2a_1/random_uniform/RandomUniform#res1b_branch2a_1/random_uniform/sub*
T0
y
res1b_branch2a_1/random_uniformAdd#res1b_branch2a_1/random_uniform/mul#res1b_branch2a_1/random_uniform/min*
T0
m
res1b_branch2a_1/kernel
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 
Ą
res1b_branch2a_1/kernel/AssignAssignres1b_branch2a_1/kernelres1b_branch2a_1/random_uniform*
use_locking(*
T0**
_class 
loc:@res1b_branch2a_1/kernel*
validate_shape(
v
res1b_branch2a_1/kernel/readIdentityres1b_branch2a_1/kernel*
T0**
_class 
loc:@res1b_branch2a_1/kernel
H
res1b_branch2a_1/ConstConst*
valueBŲ*    *
dtype0
f
res1b_branch2a_1/bias
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 
±
res1b_branch2a_1/bias/AssignAssignres1b_branch2a_1/biasres1b_branch2a_1/Const*
T0*(
_class
loc:@res1b_branch2a_1/bias*
validate_shape(*
use_locking(
p
res1b_branch2a_1/bias/readIdentityres1b_branch2a_1/bias*
T0*(
_class
loc:@res1b_branch2a_1/bias

res1b_branch2a_1/MatMulMatMulactivation_6/Relures1b_branch2a_1/kernel/read*
T0*
transpose_a( *
transpose_b( 
x
res1b_branch2a_1/BiasAddBiasAddres1b_branch2a_1/MatMulres1b_branch2a_1/bias/read*
T0*
data_formatNHWC
<
activation_7/ReluRelures1b_branch2a_1/BiasAdd*
T0
Z
%res1b_branch2b_1/random_uniform/shapeConst*
valueB"X  X  *
dtype0
P
#res1b_branch2b_1/random_uniform/minConst*
valueB
 *ĆŠ½*
dtype0
P
#res1b_branch2b_1/random_uniform/maxConst*
valueB
 *ĆŠ=*
dtype0

-res1b_branch2b_1/random_uniform/RandomUniformRandomUniform%res1b_branch2b_1/random_uniform/shape*
dtype0*
seed2É*
seed±’å)*
T0
}
#res1b_branch2b_1/random_uniform/subSub#res1b_branch2b_1/random_uniform/max#res1b_branch2b_1/random_uniform/min*
T0

#res1b_branch2b_1/random_uniform/mulMul-res1b_branch2b_1/random_uniform/RandomUniform#res1b_branch2b_1/random_uniform/sub*
T0
y
res1b_branch2b_1/random_uniformAdd#res1b_branch2b_1/random_uniform/mul#res1b_branch2b_1/random_uniform/min*
T0
m
res1b_branch2b_1/kernel
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 
Ą
res1b_branch2b_1/kernel/AssignAssignres1b_branch2b_1/kernelres1b_branch2b_1/random_uniform*
use_locking(*
T0**
_class 
loc:@res1b_branch2b_1/kernel*
validate_shape(
v
res1b_branch2b_1/kernel/readIdentityres1b_branch2b_1/kernel**
_class 
loc:@res1b_branch2b_1/kernel*
T0
H
res1b_branch2b_1/ConstConst*
dtype0*
valueBŲ*    
f
res1b_branch2b_1/bias
VariableV2*
shape:Ų*
shared_name *
dtype0*
	container 
±
res1b_branch2b_1/bias/AssignAssignres1b_branch2b_1/biasres1b_branch2b_1/Const*
use_locking(*
T0*(
_class
loc:@res1b_branch2b_1/bias*
validate_shape(
p
res1b_branch2b_1/bias/readIdentityres1b_branch2b_1/bias*
T0*(
_class
loc:@res1b_branch2b_1/bias

res1b_branch2b_1/MatMulMatMulactivation_7/Relures1b_branch2b_1/kernel/read*
transpose_b( *
T0*
transpose_a( 
x
res1b_branch2b_1/BiasAddBiasAddres1b_branch2b_1/MatMulres1b_branch2b_1/bias/read*
T0*
data_formatNHWC
H
add_4_1/addAddres1b_branch2b_1/BiasAddactivation_6/Relu*
T0
/
activation_8/ReluReluadd_4_1/add*
T0
X
#res1c_branch2a/random_uniform/shapeConst*
valueB"X  X  *
dtype0
N
!res1c_branch2a/random_uniform/minConst*
valueB
 *ĆŠ½*
dtype0
N
!res1c_branch2a/random_uniform/maxConst*
valueB
 *ĆŠ=*
dtype0

+res1c_branch2a/random_uniform/RandomUniformRandomUniform#res1c_branch2a/random_uniform/shape*
dtype0*
seed2ē*
seed±’å)*
T0
w
!res1c_branch2a/random_uniform/subSub!res1c_branch2a/random_uniform/max!res1c_branch2a/random_uniform/min*
T0

!res1c_branch2a/random_uniform/mulMul+res1c_branch2a/random_uniform/RandomUniform!res1c_branch2a/random_uniform/sub*
T0
s
res1c_branch2a/random_uniformAdd!res1c_branch2a/random_uniform/mul!res1c_branch2a/random_uniform/min*
T0
k
res1c_branch2a/kernel
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 
ø
res1c_branch2a/kernel/AssignAssignres1c_branch2a/kernelres1c_branch2a/random_uniform*
T0*(
_class
loc:@res1c_branch2a/kernel*
validate_shape(*
use_locking(
p
res1c_branch2a/kernel/readIdentityres1c_branch2a/kernel*
T0*(
_class
loc:@res1c_branch2a/kernel
F
res1c_branch2a/ConstConst*
valueBŲ*    *
dtype0
d
res1c_branch2a/bias
VariableV2*
shared_name *
dtype0*
	container *
shape:Ų
©
res1c_branch2a/bias/AssignAssignres1c_branch2a/biasres1c_branch2a/Const*
validate_shape(*
use_locking(*
T0*&
_class
loc:@res1c_branch2a/bias
j
res1c_branch2a/bias/readIdentityres1c_branch2a/bias*
T0*&
_class
loc:@res1c_branch2a/bias
}
res1c_branch2a/MatMulMatMulactivation_8/Relures1c_branch2a/kernel/read*
T0*
transpose_a( *
transpose_b( 
r
res1c_branch2a/BiasAddBiasAddres1c_branch2a/MatMulres1c_branch2a/bias/read*
T0*
data_formatNHWC
:
activation_9/ReluRelures1c_branch2a/BiasAdd*
T0
X
#res1c_branch2b/random_uniform/shapeConst*
valueB"X  X  *
dtype0
N
!res1c_branch2b/random_uniform/minConst*
valueB
 *ĆŠ½*
dtype0
N
!res1c_branch2b/random_uniform/maxConst*
valueB
 *ĆŠ=*
dtype0

+res1c_branch2b/random_uniform/RandomUniformRandomUniform#res1c_branch2b/random_uniform/shape*
T0*
dtype0*
seed2Į°*
seed±’å)
w
!res1c_branch2b/random_uniform/subSub!res1c_branch2b/random_uniform/max!res1c_branch2b/random_uniform/min*
T0

!res1c_branch2b/random_uniform/mulMul+res1c_branch2b/random_uniform/RandomUniform!res1c_branch2b/random_uniform/sub*
T0
s
res1c_branch2b/random_uniformAdd!res1c_branch2b/random_uniform/mul!res1c_branch2b/random_uniform/min*
T0
k
res1c_branch2b/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape:
ŲŲ
ø
res1c_branch2b/kernel/AssignAssignres1c_branch2b/kernelres1c_branch2b/random_uniform*
use_locking(*
T0*(
_class
loc:@res1c_branch2b/kernel*
validate_shape(
p
res1c_branch2b/kernel/readIdentityres1c_branch2b/kernel*(
_class
loc:@res1c_branch2b/kernel*
T0
F
res1c_branch2b/ConstConst*
valueBŲ*    *
dtype0
d
res1c_branch2b/bias
VariableV2*
shared_name *
dtype0*
	container *
shape:Ų
©
res1c_branch2b/bias/AssignAssignres1c_branch2b/biasres1c_branch2b/Const*
use_locking(*
T0*&
_class
loc:@res1c_branch2b/bias*
validate_shape(
j
res1c_branch2b/bias/readIdentityres1c_branch2b/bias*
T0*&
_class
loc:@res1c_branch2b/bias
}
res1c_branch2b/MatMulMatMulactivation_9/Relures1c_branch2b/kernel/read*
T0*
transpose_a( *
transpose_b( 
r
res1c_branch2b/BiasAddBiasAddres1c_branch2b/MatMulres1c_branch2b/bias/read*
T0*
data_formatNHWC
F
add_5_1/addAddres1c_branch2b/BiasAddactivation_8/Relu*
T0
0
activation_10/ReluReluadd_5_1/add*
T0
X
#res1d_branch2a/random_uniform/shapeConst*
dtype0*
valueB"X  X  
N
!res1d_branch2a/random_uniform/minConst*
valueB
 *ĆŠ½*
dtype0
N
!res1d_branch2a/random_uniform/maxConst*
dtype0*
valueB
 *ĆŠ=

+res1d_branch2a/random_uniform/RandomUniformRandomUniform#res1d_branch2a/random_uniform/shape*
seed±’å)*
T0*
dtype0*
seed2¾Ē
w
!res1d_branch2a/random_uniform/subSub!res1d_branch2a/random_uniform/max!res1d_branch2a/random_uniform/min*
T0

!res1d_branch2a/random_uniform/mulMul+res1d_branch2a/random_uniform/RandomUniform!res1d_branch2a/random_uniform/sub*
T0
s
res1d_branch2a/random_uniformAdd!res1d_branch2a/random_uniform/mul!res1d_branch2a/random_uniform/min*
T0
k
res1d_branch2a/kernel
VariableV2*
	container *
shape:
ŲŲ*
shared_name *
dtype0
ø
res1d_branch2a/kernel/AssignAssignres1d_branch2a/kernelres1d_branch2a/random_uniform*
use_locking(*
T0*(
_class
loc:@res1d_branch2a/kernel*
validate_shape(
p
res1d_branch2a/kernel/readIdentityres1d_branch2a/kernel*
T0*(
_class
loc:@res1d_branch2a/kernel
F
res1d_branch2a/ConstConst*
valueBŲ*    *
dtype0
d
res1d_branch2a/bias
VariableV2*
shared_name *
dtype0*
	container *
shape:Ų
©
res1d_branch2a/bias/AssignAssignres1d_branch2a/biasres1d_branch2a/Const*&
_class
loc:@res1d_branch2a/bias*
validate_shape(*
use_locking(*
T0
j
res1d_branch2a/bias/readIdentityres1d_branch2a/bias*&
_class
loc:@res1d_branch2a/bias*
T0
~
res1d_branch2a/MatMulMatMulactivation_10/Relures1d_branch2a/kernel/read*
T0*
transpose_a( *
transpose_b( 
r
res1d_branch2a/BiasAddBiasAddres1d_branch2a/MatMulres1d_branch2a/bias/read*
data_formatNHWC*
T0
;
activation_11/ReluRelures1d_branch2a/BiasAdd*
T0
X
#res1d_branch2b/random_uniform/shapeConst*
valueB"X  X  *
dtype0
N
!res1d_branch2b/random_uniform/minConst*
valueB
 *ĆŠ½*
dtype0
N
!res1d_branch2b/random_uniform/maxConst*
valueB
 *ĆŠ=*
dtype0

+res1d_branch2b/random_uniform/RandomUniformRandomUniform#res1d_branch2b/random_uniform/shape*
T0*
dtype0*
seed2Źńg*
seed±’å)
w
!res1d_branch2b/random_uniform/subSub!res1d_branch2b/random_uniform/max!res1d_branch2b/random_uniform/min*
T0

!res1d_branch2b/random_uniform/mulMul+res1d_branch2b/random_uniform/RandomUniform!res1d_branch2b/random_uniform/sub*
T0
s
res1d_branch2b/random_uniformAdd!res1d_branch2b/random_uniform/mul!res1d_branch2b/random_uniform/min*
T0
k
res1d_branch2b/kernel
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 
ø
res1d_branch2b/kernel/AssignAssignres1d_branch2b/kernelres1d_branch2b/random_uniform*(
_class
loc:@res1d_branch2b/kernel*
validate_shape(*
use_locking(*
T0
p
res1d_branch2b/kernel/readIdentityres1d_branch2b/kernel*
T0*(
_class
loc:@res1d_branch2b/kernel
F
res1d_branch2b/ConstConst*
valueBŲ*    *
dtype0
d
res1d_branch2b/bias
VariableV2*
shared_name *
dtype0*
	container *
shape:Ų
©
res1d_branch2b/bias/AssignAssignres1d_branch2b/biasres1d_branch2b/Const*
use_locking(*
T0*&
_class
loc:@res1d_branch2b/bias*
validate_shape(
j
res1d_branch2b/bias/readIdentityres1d_branch2b/bias*
T0*&
_class
loc:@res1d_branch2b/bias
~
res1d_branch2b/MatMulMatMulactivation_11/Relures1d_branch2b/kernel/read*
transpose_a( *
transpose_b( *
T0
r
res1d_branch2b/BiasAddBiasAddres1d_branch2b/MatMulres1d_branch2b/bias/read*
T0*
data_formatNHWC
G
add_6_1/addAddres1d_branch2b/BiasAddactivation_10/Relu*
T0
0
activation_12/ReluReluadd_6_1/add*
T0
X
#res1e_branch2a/random_uniform/shapeConst*
valueB"X  X  *
dtype0
N
!res1e_branch2a/random_uniform/minConst*
valueB
 *ĆŠ½*
dtype0
N
!res1e_branch2a/random_uniform/maxConst*
dtype0*
valueB
 *ĆŠ=

+res1e_branch2a/random_uniform/RandomUniformRandomUniform#res1e_branch2a/random_uniform/shape*
T0*
dtype0*
seed2­*
seed±’å)
w
!res1e_branch2a/random_uniform/subSub!res1e_branch2a/random_uniform/max!res1e_branch2a/random_uniform/min*
T0

!res1e_branch2a/random_uniform/mulMul+res1e_branch2a/random_uniform/RandomUniform!res1e_branch2a/random_uniform/sub*
T0
s
res1e_branch2a/random_uniformAdd!res1e_branch2a/random_uniform/mul!res1e_branch2a/random_uniform/min*
T0
k
res1e_branch2a/kernel
VariableV2*
shape:
ŲŲ*
shared_name *
dtype0*
	container 
ø
res1e_branch2a/kernel/AssignAssignres1e_branch2a/kernelres1e_branch2a/random_uniform*
use_locking(*
T0*(
_class
loc:@res1e_branch2a/kernel*
validate_shape(
p
res1e_branch2a/kernel/readIdentityres1e_branch2a/kernel*(
_class
loc:@res1e_branch2a/kernel*
T0
F
res1e_branch2a/ConstConst*
valueBŲ*    *
dtype0
d
res1e_branch2a/bias
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 
©
res1e_branch2a/bias/AssignAssignres1e_branch2a/biasres1e_branch2a/Const*
use_locking(*
T0*&
_class
loc:@res1e_branch2a/bias*
validate_shape(
j
res1e_branch2a/bias/readIdentityres1e_branch2a/bias*
T0*&
_class
loc:@res1e_branch2a/bias
~
res1e_branch2a/MatMulMatMulactivation_12/Relures1e_branch2a/kernel/read*
transpose_a( *
transpose_b( *
T0
r
res1e_branch2a/BiasAddBiasAddres1e_branch2a/MatMulres1e_branch2a/bias/read*
T0*
data_formatNHWC
;
activation_13/ReluRelures1e_branch2a/BiasAdd*
T0
X
#res1e_branch2b/random_uniform/shapeConst*
valueB"X  X  *
dtype0
N
!res1e_branch2b/random_uniform/minConst*
valueB
 *ĆŠ½*
dtype0
N
!res1e_branch2b/random_uniform/maxConst*
valueB
 *ĆŠ=*
dtype0

+res1e_branch2b/random_uniform/RandomUniformRandomUniform#res1e_branch2b/random_uniform/shape*
dtype0*
seed2ū*
seed±’å)*
T0
w
!res1e_branch2b/random_uniform/subSub!res1e_branch2b/random_uniform/max!res1e_branch2b/random_uniform/min*
T0

!res1e_branch2b/random_uniform/mulMul+res1e_branch2b/random_uniform/RandomUniform!res1e_branch2b/random_uniform/sub*
T0
s
res1e_branch2b/random_uniformAdd!res1e_branch2b/random_uniform/mul!res1e_branch2b/random_uniform/min*
T0
k
res1e_branch2b/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape:
ŲŲ
ø
res1e_branch2b/kernel/AssignAssignres1e_branch2b/kernelres1e_branch2b/random_uniform*(
_class
loc:@res1e_branch2b/kernel*
validate_shape(*
use_locking(*
T0
p
res1e_branch2b/kernel/readIdentityres1e_branch2b/kernel*(
_class
loc:@res1e_branch2b/kernel*
T0
F
res1e_branch2b/ConstConst*
valueBŲ*    *
dtype0
d
res1e_branch2b/bias
VariableV2*
shape:Ų*
shared_name *
dtype0*
	container 
©
res1e_branch2b/bias/AssignAssignres1e_branch2b/biasres1e_branch2b/Const*
use_locking(*
T0*&
_class
loc:@res1e_branch2b/bias*
validate_shape(
j
res1e_branch2b/bias/readIdentityres1e_branch2b/bias*
T0*&
_class
loc:@res1e_branch2b/bias
~
res1e_branch2b/MatMulMatMulactivation_13/Relures1e_branch2b/kernel/read*
transpose_a( *
transpose_b( *
T0
r
res1e_branch2b/BiasAddBiasAddres1e_branch2b/MatMulres1e_branch2b/bias/read*
T0*
data_formatNHWC
G
add_7_1/addAddres1e_branch2b/BiasAddactivation_12/Relu*
T0
0
activation_14/ReluReluadd_7_1/add*
T0
Q
dense_4/random_uniform/shapeConst*
valueB"X     *
dtype0
G
dense_4/random_uniform/minConst*
valueB
 *Ė½*
dtype0
G
dense_4/random_uniform/maxConst*
valueB
 *Ė=*
dtype0

$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
T0*
dtype0*
seed2Ŗž*
seed±’å)
b
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0
l
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0
^
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0
c
dense_4/kernel
VariableV2*
dtype0*
	container *
shape:	Ų*
shared_name 

dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
validate_shape(*
use_locking(*
T0*!
_class
loc:@dense_4/kernel
[
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel
>
dense_4/ConstConst*
valueB*    *
dtype0
\
dense_4/bias
VariableV2*
shared_name *
dtype0*
	container *
shape:

dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(
U
dense_4/bias/readIdentitydense_4/bias*
_class
loc:@dense_4/bias*
T0
p
dense_4/MatMulMatMulactivation_14/Reludense_4/kernel/read*
transpose_a( *
transpose_b( *
T0
]
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
T0*
data_formatNHWC
G
iterations_1/initial_valueConst*
valueB
 *    *
dtype0
X
iterations_1
VariableV2*
shared_name *
dtype0*
	container *
shape: 

iterations_1/AssignAssigniterations_1iterations_1/initial_value*
use_locking(*
T0*
_class
loc:@iterations_1*
validate_shape(
U
iterations_1/readIdentityiterations_1*
T0*
_class
loc:@iterations_1
?
lr_1/initial_valueConst*
valueB
 *o:*
dtype0
P
lr_1
VariableV2*
shape: *
shared_name *
dtype0*
	container 
z
lr_1/AssignAssignlr_1lr_1/initial_value*
use_locking(*
T0*
_class
	loc:@lr_1*
validate_shape(
=
	lr_1/readIdentitylr_1*
T0*
_class
	loc:@lr_1
C
beta_1_1/initial_valueConst*
valueB
 *fff?*
dtype0
T
beta_1_1
VariableV2*
dtype0*
	container *
shape: *
shared_name 

beta_1_1/AssignAssignbeta_1_1beta_1_1/initial_value*
use_locking(*
T0*
_class
loc:@beta_1_1*
validate_shape(
I
beta_1_1/readIdentitybeta_1_1*
T0*
_class
loc:@beta_1_1
C
beta_2_1/initial_valueConst*
valueB
 *w¾?*
dtype0
T
beta_2_1
VariableV2*
dtype0*
	container *
shape: *
shared_name 

beta_2_1/AssignAssignbeta_2_1beta_2_1/initial_value*
validate_shape(*
use_locking(*
T0*
_class
loc:@beta_2_1
I
beta_2_1/readIdentitybeta_2_1*
T0*
_class
loc:@beta_2_1
B
decay_1/initial_valueConst*
valueB
 *    *
dtype0
S
decay_1
VariableV2*
dtype0*
	container *
shape: *
shared_name 

decay_1/AssignAssigndecay_1decay_1/initial_value*
use_locking(*
T0*
_class
loc:@decay_1*
validate_shape(
F
decay_1/readIdentitydecay_1*
T0*
_class
loc:@decay_1
L
dense_4_sample_weightsPlaceholder*
dtype0*
shape:’’’’’’’’’
Q
dense_4_targetPlaceholder*
dtype0*%
shape:’’’’’’’’’’’’’’’’’’
7
sub_39Subdense_4/BiasAdddense_4_target*
T0
$
	Square_13Squaresub_39*
T0
B
Mean_5/reduction_indicesConst*
value	B :*
dtype0
Y
Mean_5Mean	Square_13Mean_5/reduction_indices*

Tidx0*
	keep_dims( *
T0
A
Mean_6/reduction_indicesConst*
valueB *
dtype0
V
Mean_6MeanMean_5Mean_6/reduction_indices*

Tidx0*
	keep_dims( *
T0
6
mul_63MulMean_6dense_4_sample_weights*
T0
9
NotEqual_1/yConst*
valueB
 *    *
dtype0
E

NotEqual_1NotEqualdense_4_sample_weightsNotEqual_1/y*
T0
B
Cast_2Cast
NotEqual_1*

SrcT0
*
Truncate( *

DstT0
6
Const_53Const*
valueB: *
dtype0
F
Mean_7MeanCast_2Const_53*
T0*

Tidx0*
	keep_dims( 
.

truediv_14RealDivmul_63Mean_7*
T0
6
Const_54Const*
valueB: *
dtype0
J
Mean_8Mean
truediv_14Const_54*
T0*

Tidx0*
	keep_dims( 
5
mul_64/xConst*
valueB
 *  ?*
dtype0
(
mul_64Mulmul_64/xMean_8*
T0
<
ArgMax_2/dimensionConst*
value	B :*
dtype0
^
ArgMax_2ArgMaxdense_4_targetArgMax_2/dimension*
T0*
output_type0	*

Tidx0
<
ArgMax_3/dimensionConst*
value	B :*
dtype0
_
ArgMax_3ArgMaxdense_4/BiasAddArgMax_3/dimension*
T0*
output_type0	*

Tidx0
-
Equal_1EqualArgMax_2ArgMax_3*
T0	
?
Cast_3CastEqual_1*

SrcT0
*
Truncate( *

DstT0
6
Const_55Const*
valueB: *
dtype0
F
Mean_9MeanCast_3Const_55*
T0*

Tidx0*
	keep_dims( 
&
group_deps_3NoOp^Mean_9^mul_64
U
gradients_1/ShapeConst*
valueB *
_class
loc:@mul_64*
dtype0
]
gradients_1/grad_ys_0Const*
valueB
 *  ?*
_class
loc:@mul_64*
dtype0
x
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_class
loc:@mul_64
`
gradients_1/mul_64_grad/MulMulgradients_1/FillMean_8*
T0*
_class
loc:@mul_64
d
gradients_1/mul_64_grad/Mul_1Mulgradients_1/Fillmul_64/x*
_class
loc:@mul_64*
T0
n
%gradients_1/Mean_8_grad/Reshape/shapeConst*
valueB:*
_class
loc:@Mean_8*
dtype0
¢
gradients_1/Mean_8_grad/ReshapeReshapegradients_1/mul_64_grad/Mul_1%gradients_1/Mean_8_grad/Reshape/shape*
T0*
Tshape0*
_class
loc:@Mean_8
f
gradients_1/Mean_8_grad/ShapeShape
truediv_14*
T0*
out_type0*
_class
loc:@Mean_8

gradients_1/Mean_8_grad/TileTilegradients_1/Mean_8_grad/Reshapegradients_1/Mean_8_grad/Shape*

Tmultiples0*
T0*
_class
loc:@Mean_8
h
gradients_1/Mean_8_grad/Shape_1Shape
truediv_14*
T0*
out_type0*
_class
loc:@Mean_8
c
gradients_1/Mean_8_grad/Shape_2Const*
valueB *
_class
loc:@Mean_8*
dtype0
f
gradients_1/Mean_8_grad/ConstConst*
valueB: *
_class
loc:@Mean_8*
dtype0
„
gradients_1/Mean_8_grad/ProdProdgradients_1/Mean_8_grad/Shape_1gradients_1/Mean_8_grad/Const*
T0*
_class
loc:@Mean_8*

Tidx0*
	keep_dims( 
h
gradients_1/Mean_8_grad/Const_1Const*
valueB: *
_class
loc:@Mean_8*
dtype0
©
gradients_1/Mean_8_grad/Prod_1Prodgradients_1/Mean_8_grad/Shape_2gradients_1/Mean_8_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_class
loc:@Mean_8
f
!gradients_1/Mean_8_grad/Maximum/yConst*
value	B :*
_class
loc:@Mean_8*
dtype0

gradients_1/Mean_8_grad/MaximumMaximumgradients_1/Mean_8_grad/Prod_1!gradients_1/Mean_8_grad/Maximum/y*
_class
loc:@Mean_8*
T0

 gradients_1/Mean_8_grad/floordivFloorDivgradients_1/Mean_8_grad/Prodgradients_1/Mean_8_grad/Maximum*
T0*
_class
loc:@Mean_8

gradients_1/Mean_8_grad/CastCast gradients_1/Mean_8_grad/floordiv*

DstT0*

SrcT0*
_class
loc:@Mean_8*
Truncate( 

gradients_1/Mean_8_grad/truedivRealDivgradients_1/Mean_8_grad/Tilegradients_1/Mean_8_grad/Cast*
T0*
_class
loc:@Mean_8
j
!gradients_1/truediv_14_grad/ShapeShapemul_63*
T0*
out_type0*
_class
loc:@truediv_14
k
#gradients_1/truediv_14_grad/Shape_1Const*
valueB *
_class
loc:@truediv_14*
dtype0
ŗ
1gradients_1/truediv_14_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients_1/truediv_14_grad/Shape#gradients_1/truediv_14_grad/Shape_1*
T0*
_class
loc:@truediv_14

#gradients_1/truediv_14_grad/RealDivRealDivgradients_1/Mean_8_grad/truedivMean_7*
T0*
_class
loc:@truediv_14
Ć
gradients_1/truediv_14_grad/SumSum#gradients_1/truediv_14_grad/RealDiv1gradients_1/truediv_14_grad/BroadcastGradientArgs*
T0*
_class
loc:@truediv_14*

Tidx0*
	keep_dims( 
Ø
#gradients_1/truediv_14_grad/ReshapeReshapegradients_1/truediv_14_grad/Sum!gradients_1/truediv_14_grad/Shape*
Tshape0*
_class
loc:@truediv_14*
T0
V
gradients_1/truediv_14_grad/NegNegmul_63*
T0*
_class
loc:@truediv_14

%gradients_1/truediv_14_grad/RealDiv_1RealDivgradients_1/truediv_14_grad/NegMean_7*
_class
loc:@truediv_14*
T0

%gradients_1/truediv_14_grad/RealDiv_2RealDiv%gradients_1/truediv_14_grad/RealDiv_1Mean_7*
T0*
_class
loc:@truediv_14

gradients_1/truediv_14_grad/mulMulgradients_1/Mean_8_grad/truediv%gradients_1/truediv_14_grad/RealDiv_2*
T0*
_class
loc:@truediv_14
Ć
!gradients_1/truediv_14_grad/Sum_1Sumgradients_1/truediv_14_grad/mul3gradients_1/truediv_14_grad/BroadcastGradientArgs:1*
_class
loc:@truediv_14*

Tidx0*
	keep_dims( *
T0
®
%gradients_1/truediv_14_grad/Reshape_1Reshape!gradients_1/truediv_14_grad/Sum_1#gradients_1/truediv_14_grad/Shape_1*
T0*
Tshape0*
_class
loc:@truediv_14
b
gradients_1/mul_63_grad/ShapeShapeMean_6*
T0*
out_type0*
_class
loc:@mul_63
t
gradients_1/mul_63_grad/Shape_1Shapedense_4_sample_weights*
T0*
out_type0*
_class
loc:@mul_63
Ŗ
-gradients_1/mul_63_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_63_grad/Shapegradients_1/mul_63_grad/Shape_1*
T0*
_class
loc:@mul_63

gradients_1/mul_63_grad/MulMul#gradients_1/truediv_14_grad/Reshapedense_4_sample_weights*
T0*
_class
loc:@mul_63
Æ
gradients_1/mul_63_grad/SumSumgradients_1/mul_63_grad/Mul-gradients_1/mul_63_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_class
loc:@mul_63

gradients_1/mul_63_grad/ReshapeReshapegradients_1/mul_63_grad/Sumgradients_1/mul_63_grad/Shape*
T0*
Tshape0*
_class
loc:@mul_63
u
gradients_1/mul_63_grad/Mul_1MulMean_6#gradients_1/truediv_14_grad/Reshape*
T0*
_class
loc:@mul_63
µ
gradients_1/mul_63_grad/Sum_1Sumgradients_1/mul_63_grad/Mul_1/gradients_1/mul_63_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_class
loc:@mul_63

!gradients_1/mul_63_grad/Reshape_1Reshapegradients_1/mul_63_grad/Sum_1gradients_1/mul_63_grad/Shape_1*
T0*
Tshape0*
_class
loc:@mul_63
b
gradients_1/Mean_6_grad/ShapeShapeMean_5*
T0*
out_type0*
_class
loc:@Mean_6
a
gradients_1/Mean_6_grad/SizeConst*
value	B :*
_class
loc:@Mean_6*
dtype0
~
gradients_1/Mean_6_grad/addAddMean_6/reduction_indicesgradients_1/Mean_6_grad/Size*
_class
loc:@Mean_6*
T0

gradients_1/Mean_6_grad/modFloorModgradients_1/Mean_6_grad/addgradients_1/Mean_6_grad/Size*
T0*
_class
loc:@Mean_6
h
gradients_1/Mean_6_grad/Shape_1Const*
valueB: *
_class
loc:@Mean_6*
dtype0
h
#gradients_1/Mean_6_grad/range/startConst*
value	B : *
_class
loc:@Mean_6*
dtype0
h
#gradients_1/Mean_6_grad/range/deltaConst*
value	B :*
_class
loc:@Mean_6*
dtype0
µ
gradients_1/Mean_6_grad/rangeRange#gradients_1/Mean_6_grad/range/startgradients_1/Mean_6_grad/Size#gradients_1/Mean_6_grad/range/delta*
_class
loc:@Mean_6*

Tidx0
g
"gradients_1/Mean_6_grad/Fill/valueConst*
value	B :*
_class
loc:@Mean_6*
dtype0

gradients_1/Mean_6_grad/FillFillgradients_1/Mean_6_grad/Shape_1"gradients_1/Mean_6_grad/Fill/value*
T0*

index_type0*
_class
loc:@Mean_6
Ü
%gradients_1/Mean_6_grad/DynamicStitchDynamicStitchgradients_1/Mean_6_grad/rangegradients_1/Mean_6_grad/modgradients_1/Mean_6_grad/Shapegradients_1/Mean_6_grad/Fill*
T0*
_class
loc:@Mean_6*
N
f
!gradients_1/Mean_6_grad/Maximum/yConst*
dtype0*
value	B :*
_class
loc:@Mean_6

gradients_1/Mean_6_grad/MaximumMaximum%gradients_1/Mean_6_grad/DynamicStitch!gradients_1/Mean_6_grad/Maximum/y*
T0*
_class
loc:@Mean_6

 gradients_1/Mean_6_grad/floordivFloorDivgradients_1/Mean_6_grad/Shapegradients_1/Mean_6_grad/Maximum*
T0*
_class
loc:@Mean_6
¤
gradients_1/Mean_6_grad/ReshapeReshapegradients_1/mul_63_grad/Reshape%gradients_1/Mean_6_grad/DynamicStitch*
T0*
Tshape0*
_class
loc:@Mean_6

gradients_1/Mean_6_grad/TileTilegradients_1/Mean_6_grad/Reshape gradients_1/Mean_6_grad/floordiv*
_class
loc:@Mean_6*

Tmultiples0*
T0
d
gradients_1/Mean_6_grad/Shape_2ShapeMean_5*
T0*
out_type0*
_class
loc:@Mean_6
d
gradients_1/Mean_6_grad/Shape_3ShapeMean_6*
T0*
out_type0*
_class
loc:@Mean_6
f
gradients_1/Mean_6_grad/ConstConst*
valueB: *
_class
loc:@Mean_6*
dtype0
„
gradients_1/Mean_6_grad/ProdProdgradients_1/Mean_6_grad/Shape_2gradients_1/Mean_6_grad/Const*

Tidx0*
	keep_dims( *
T0*
_class
loc:@Mean_6
h
gradients_1/Mean_6_grad/Const_1Const*
valueB: *
_class
loc:@Mean_6*
dtype0
©
gradients_1/Mean_6_grad/Prod_1Prodgradients_1/Mean_6_grad/Shape_3gradients_1/Mean_6_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_class
loc:@Mean_6
h
#gradients_1/Mean_6_grad/Maximum_1/yConst*
value	B :*
_class
loc:@Mean_6*
dtype0

!gradients_1/Mean_6_grad/Maximum_1Maximumgradients_1/Mean_6_grad/Prod_1#gradients_1/Mean_6_grad/Maximum_1/y*
T0*
_class
loc:@Mean_6

"gradients_1/Mean_6_grad/floordiv_1FloorDivgradients_1/Mean_6_grad/Prod!gradients_1/Mean_6_grad/Maximum_1*
T0*
_class
loc:@Mean_6

gradients_1/Mean_6_grad/CastCast"gradients_1/Mean_6_grad/floordiv_1*

SrcT0*
_class
loc:@Mean_6*
Truncate( *

DstT0

gradients_1/Mean_6_grad/truedivRealDivgradients_1/Mean_6_grad/Tilegradients_1/Mean_6_grad/Cast*
T0*
_class
loc:@Mean_6
e
gradients_1/Mean_5_grad/ShapeShape	Square_13*
T0*
out_type0*
_class
loc:@Mean_5
a
gradients_1/Mean_5_grad/SizeConst*
value	B :*
_class
loc:@Mean_5*
dtype0
~
gradients_1/Mean_5_grad/addAddMean_5/reduction_indicesgradients_1/Mean_5_grad/Size*
T0*
_class
loc:@Mean_5

gradients_1/Mean_5_grad/modFloorModgradients_1/Mean_5_grad/addgradients_1/Mean_5_grad/Size*
T0*
_class
loc:@Mean_5
c
gradients_1/Mean_5_grad/Shape_1Const*
valueB *
_class
loc:@Mean_5*
dtype0
h
#gradients_1/Mean_5_grad/range/startConst*
value	B : *
_class
loc:@Mean_5*
dtype0
h
#gradients_1/Mean_5_grad/range/deltaConst*
value	B :*
_class
loc:@Mean_5*
dtype0
µ
gradients_1/Mean_5_grad/rangeRange#gradients_1/Mean_5_grad/range/startgradients_1/Mean_5_grad/Size#gradients_1/Mean_5_grad/range/delta*
_class
loc:@Mean_5*

Tidx0
g
"gradients_1/Mean_5_grad/Fill/valueConst*
dtype0*
value	B :*
_class
loc:@Mean_5

gradients_1/Mean_5_grad/FillFillgradients_1/Mean_5_grad/Shape_1"gradients_1/Mean_5_grad/Fill/value*
T0*

index_type0*
_class
loc:@Mean_5
Ü
%gradients_1/Mean_5_grad/DynamicStitchDynamicStitchgradients_1/Mean_5_grad/rangegradients_1/Mean_5_grad/modgradients_1/Mean_5_grad/Shapegradients_1/Mean_5_grad/Fill*
T0*
_class
loc:@Mean_5*
N
f
!gradients_1/Mean_5_grad/Maximum/yConst*
value	B :*
_class
loc:@Mean_5*
dtype0

gradients_1/Mean_5_grad/MaximumMaximum%gradients_1/Mean_5_grad/DynamicStitch!gradients_1/Mean_5_grad/Maximum/y*
_class
loc:@Mean_5*
T0

 gradients_1/Mean_5_grad/floordivFloorDivgradients_1/Mean_5_grad/Shapegradients_1/Mean_5_grad/Maximum*
T0*
_class
loc:@Mean_5
¤
gradients_1/Mean_5_grad/ReshapeReshapegradients_1/Mean_6_grad/truediv%gradients_1/Mean_5_grad/DynamicStitch*
T0*
Tshape0*
_class
loc:@Mean_5

gradients_1/Mean_5_grad/TileTilegradients_1/Mean_5_grad/Reshape gradients_1/Mean_5_grad/floordiv*

Tmultiples0*
T0*
_class
loc:@Mean_5
g
gradients_1/Mean_5_grad/Shape_2Shape	Square_13*
T0*
out_type0*
_class
loc:@Mean_5
d
gradients_1/Mean_5_grad/Shape_3ShapeMean_5*
out_type0*
_class
loc:@Mean_5*
T0
f
gradients_1/Mean_5_grad/ConstConst*
valueB: *
_class
loc:@Mean_5*
dtype0
„
gradients_1/Mean_5_grad/ProdProdgradients_1/Mean_5_grad/Shape_2gradients_1/Mean_5_grad/Const*
_class
loc:@Mean_5*

Tidx0*
	keep_dims( *
T0
h
gradients_1/Mean_5_grad/Const_1Const*
valueB: *
_class
loc:@Mean_5*
dtype0
©
gradients_1/Mean_5_grad/Prod_1Prodgradients_1/Mean_5_grad/Shape_3gradients_1/Mean_5_grad/Const_1*
T0*
_class
loc:@Mean_5*

Tidx0*
	keep_dims( 
h
#gradients_1/Mean_5_grad/Maximum_1/yConst*
value	B :*
_class
loc:@Mean_5*
dtype0

!gradients_1/Mean_5_grad/Maximum_1Maximumgradients_1/Mean_5_grad/Prod_1#gradients_1/Mean_5_grad/Maximum_1/y*
_class
loc:@Mean_5*
T0

"gradients_1/Mean_5_grad/floordiv_1FloorDivgradients_1/Mean_5_grad/Prod!gradients_1/Mean_5_grad/Maximum_1*
T0*
_class
loc:@Mean_5

gradients_1/Mean_5_grad/CastCast"gradients_1/Mean_5_grad/floordiv_1*
_class
loc:@Mean_5*
Truncate( *

DstT0*

SrcT0

gradients_1/Mean_5_grad/truedivRealDivgradients_1/Mean_5_grad/Tilegradients_1/Mean_5_grad/Cast*
T0*
_class
loc:@Mean_5

 gradients_1/Square_13_grad/ConstConst ^gradients_1/Mean_5_grad/truediv*
dtype0*
valueB
 *   @*
_class
loc:@Square_13
v
gradients_1/Square_13_grad/MulMulsub_39 gradients_1/Square_13_grad/Const*
T0*
_class
loc:@Square_13

 gradients_1/Square_13_grad/Mul_1Mulgradients_1/Mean_5_grad/truedivgradients_1/Square_13_grad/Mul*
T0*
_class
loc:@Square_13
k
gradients_1/sub_39_grad/ShapeShapedense_4/BiasAdd*
T0*
out_type0*
_class
loc:@sub_39
l
gradients_1/sub_39_grad/Shape_1Shapedense_4_target*
T0*
out_type0*
_class
loc:@sub_39
Ŗ
-gradients_1/sub_39_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_39_grad/Shapegradients_1/sub_39_grad/Shape_1*
T0*
_class
loc:@sub_39
“
gradients_1/sub_39_grad/SumSum gradients_1/Square_13_grad/Mul_1-gradients_1/sub_39_grad/BroadcastGradientArgs*
T0*
_class
loc:@sub_39*

Tidx0*
	keep_dims( 

gradients_1/sub_39_grad/ReshapeReshapegradients_1/sub_39_grad/Sumgradients_1/sub_39_grad/Shape*
T0*
Tshape0*
_class
loc:@sub_39
ø
gradients_1/sub_39_grad/Sum_1Sum gradients_1/Square_13_grad/Mul_1/gradients_1/sub_39_grad/BroadcastGradientArgs:1*
T0*
_class
loc:@sub_39*

Tidx0*
	keep_dims( 
e
gradients_1/sub_39_grad/NegNeggradients_1/sub_39_grad/Sum_1*
_class
loc:@sub_39*
T0

!gradients_1/sub_39_grad/Reshape_1Reshapegradients_1/sub_39_grad/Neggradients_1/sub_39_grad/Shape_1*
T0*
Tshape0*
_class
loc:@sub_39
 
,gradients_1/dense_4/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/sub_39_grad/Reshape*
data_formatNHWC*
T0*"
_class
loc:@dense_4/BiasAdd
ø
&gradients_1/dense_4/MatMul_grad/MatMulMatMulgradients_1/sub_39_grad/Reshapedense_4/kernel/read*
transpose_b(*
T0*!
_class
loc:@dense_4/MatMul*
transpose_a( 
¹
(gradients_1/dense_4/MatMul_grad/MatMul_1MatMulactivation_14/Relugradients_1/sub_39_grad/Reshape*
T0*!
_class
loc:@dense_4/MatMul*
transpose_a(*
transpose_b( 
¤
,gradients_1/activation_14/Relu_grad/ReluGradReluGrad&gradients_1/dense_4/MatMul_grad/MatMulactivation_14/Relu*
T0*%
_class
loc:@activation_14/Relu
|
"gradients_1/add_7_1/add_grad/ShapeShaperes1e_branch2b/BiasAdd*
T0*
out_type0*
_class
loc:@add_7_1/add
z
$gradients_1/add_7_1/add_grad/Shape_1Shapeactivation_12/Relu*
T0*
out_type0*
_class
loc:@add_7_1/add
¾
2gradients_1/add_7_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_1/add_7_1/add_grad/Shape$gradients_1/add_7_1/add_grad/Shape_1*
T0*
_class
loc:@add_7_1/add
Ļ
 gradients_1/add_7_1/add_grad/SumSum,gradients_1/activation_14/Relu_grad/ReluGrad2gradients_1/add_7_1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_class
loc:@add_7_1/add
¬
$gradients_1/add_7_1/add_grad/ReshapeReshape gradients_1/add_7_1/add_grad/Sum"gradients_1/add_7_1/add_grad/Shape*
T0*
Tshape0*
_class
loc:@add_7_1/add
Ó
"gradients_1/add_7_1/add_grad/Sum_1Sum,gradients_1/activation_14/Relu_grad/ReluGrad4gradients_1/add_7_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_class
loc:@add_7_1/add
²
&gradients_1/add_7_1/add_grad/Reshape_1Reshape"gradients_1/add_7_1/add_grad/Sum_1$gradients_1/add_7_1/add_grad/Shape_1*
T0*
Tshape0*
_class
loc:@add_7_1/add
³
3gradients_1/res1e_branch2b/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients_1/add_7_1/add_grad/Reshape*
T0*)
_class
loc:@res1e_branch2b/BiasAdd*
data_formatNHWC
Ņ
-gradients_1/res1e_branch2b/MatMul_grad/MatMulMatMul$gradients_1/add_7_1/add_grad/Reshaperes1e_branch2b/kernel/read*
transpose_b(*
T0*(
_class
loc:@res1e_branch2b/MatMul*
transpose_a( 
Ģ
/gradients_1/res1e_branch2b/MatMul_grad/MatMul_1MatMulactivation_13/Relu$gradients_1/add_7_1/add_grad/Reshape*
T0*(
_class
loc:@res1e_branch2b/MatMul*
transpose_a(*
transpose_b( 
«
,gradients_1/activation_13/Relu_grad/ReluGradReluGrad-gradients_1/res1e_branch2b/MatMul_grad/MatMulactivation_13/Relu*
T0*%
_class
loc:@activation_13/Relu
»
3gradients_1/res1e_branch2a/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_1/activation_13/Relu_grad/ReluGrad*
T0*)
_class
loc:@res1e_branch2a/BiasAdd*
data_formatNHWC
Ś
-gradients_1/res1e_branch2a/MatMul_grad/MatMulMatMul,gradients_1/activation_13/Relu_grad/ReluGradres1e_branch2a/kernel/read*
transpose_b(*
T0*(
_class
loc:@res1e_branch2a/MatMul*
transpose_a( 
Ō
/gradients_1/res1e_branch2a/MatMul_grad/MatMul_1MatMulactivation_12/Relu,gradients_1/activation_13/Relu_grad/ReluGrad*
T0*(
_class
loc:@res1e_branch2a/MatMul*
transpose_a(*
transpose_b( 
”
gradients_1/AddNAddN&gradients_1/add_7_1/add_grad/Reshape_1-gradients_1/res1e_branch2a/MatMul_grad/MatMul*
T0*
_class
loc:@add_7_1/add*
N

,gradients_1/activation_12/Relu_grad/ReluGradReluGradgradients_1/AddNactivation_12/Relu*
T0*%
_class
loc:@activation_12/Relu
|
"gradients_1/add_6_1/add_grad/ShapeShaperes1d_branch2b/BiasAdd*
T0*
out_type0*
_class
loc:@add_6_1/add
z
$gradients_1/add_6_1/add_grad/Shape_1Shapeactivation_10/Relu*
T0*
out_type0*
_class
loc:@add_6_1/add
¾
2gradients_1/add_6_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_1/add_6_1/add_grad/Shape$gradients_1/add_6_1/add_grad/Shape_1*
T0*
_class
loc:@add_6_1/add
Ļ
 gradients_1/add_6_1/add_grad/SumSum,gradients_1/activation_12/Relu_grad/ReluGrad2gradients_1/add_6_1/add_grad/BroadcastGradientArgs*
T0*
_class
loc:@add_6_1/add*

Tidx0*
	keep_dims( 
¬
$gradients_1/add_6_1/add_grad/ReshapeReshape gradients_1/add_6_1/add_grad/Sum"gradients_1/add_6_1/add_grad/Shape*
T0*
Tshape0*
_class
loc:@add_6_1/add
Ó
"gradients_1/add_6_1/add_grad/Sum_1Sum,gradients_1/activation_12/Relu_grad/ReluGrad4gradients_1/add_6_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_class
loc:@add_6_1/add
²
&gradients_1/add_6_1/add_grad/Reshape_1Reshape"gradients_1/add_6_1/add_grad/Sum_1$gradients_1/add_6_1/add_grad/Shape_1*
T0*
Tshape0*
_class
loc:@add_6_1/add
³
3gradients_1/res1d_branch2b/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients_1/add_6_1/add_grad/Reshape*
T0*)
_class
loc:@res1d_branch2b/BiasAdd*
data_formatNHWC
Ņ
-gradients_1/res1d_branch2b/MatMul_grad/MatMulMatMul$gradients_1/add_6_1/add_grad/Reshaperes1d_branch2b/kernel/read*
T0*(
_class
loc:@res1d_branch2b/MatMul*
transpose_a( *
transpose_b(
Ģ
/gradients_1/res1d_branch2b/MatMul_grad/MatMul_1MatMulactivation_11/Relu$gradients_1/add_6_1/add_grad/Reshape*
T0*(
_class
loc:@res1d_branch2b/MatMul*
transpose_a(*
transpose_b( 
«
,gradients_1/activation_11/Relu_grad/ReluGradReluGrad-gradients_1/res1d_branch2b/MatMul_grad/MatMulactivation_11/Relu*
T0*%
_class
loc:@activation_11/Relu
»
3gradients_1/res1d_branch2a/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_1/activation_11/Relu_grad/ReluGrad*)
_class
loc:@res1d_branch2a/BiasAdd*
data_formatNHWC*
T0
Ś
-gradients_1/res1d_branch2a/MatMul_grad/MatMulMatMul,gradients_1/activation_11/Relu_grad/ReluGradres1d_branch2a/kernel/read*
transpose_a( *
transpose_b(*
T0*(
_class
loc:@res1d_branch2a/MatMul
Ō
/gradients_1/res1d_branch2a/MatMul_grad/MatMul_1MatMulactivation_10/Relu,gradients_1/activation_11/Relu_grad/ReluGrad*(
_class
loc:@res1d_branch2a/MatMul*
transpose_a(*
transpose_b( *
T0
£
gradients_1/AddN_1AddN&gradients_1/add_6_1/add_grad/Reshape_1-gradients_1/res1d_branch2a/MatMul_grad/MatMul*
T0*
_class
loc:@add_6_1/add*
N

,gradients_1/activation_10/Relu_grad/ReluGradReluGradgradients_1/AddN_1activation_10/Relu*
T0*%
_class
loc:@activation_10/Relu
|
"gradients_1/add_5_1/add_grad/ShapeShaperes1c_branch2b/BiasAdd*
T0*
out_type0*
_class
loc:@add_5_1/add
y
$gradients_1/add_5_1/add_grad/Shape_1Shapeactivation_8/Relu*
out_type0*
_class
loc:@add_5_1/add*
T0
¾
2gradients_1/add_5_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_1/add_5_1/add_grad/Shape$gradients_1/add_5_1/add_grad/Shape_1*
T0*
_class
loc:@add_5_1/add
Ļ
 gradients_1/add_5_1/add_grad/SumSum,gradients_1/activation_10/Relu_grad/ReluGrad2gradients_1/add_5_1/add_grad/BroadcastGradientArgs*
_class
loc:@add_5_1/add*

Tidx0*
	keep_dims( *
T0
¬
$gradients_1/add_5_1/add_grad/ReshapeReshape gradients_1/add_5_1/add_grad/Sum"gradients_1/add_5_1/add_grad/Shape*
T0*
Tshape0*
_class
loc:@add_5_1/add
Ó
"gradients_1/add_5_1/add_grad/Sum_1Sum,gradients_1/activation_10/Relu_grad/ReluGrad4gradients_1/add_5_1/add_grad/BroadcastGradientArgs:1*
T0*
_class
loc:@add_5_1/add*

Tidx0*
	keep_dims( 
²
&gradients_1/add_5_1/add_grad/Reshape_1Reshape"gradients_1/add_5_1/add_grad/Sum_1$gradients_1/add_5_1/add_grad/Shape_1*
T0*
Tshape0*
_class
loc:@add_5_1/add
³
3gradients_1/res1c_branch2b/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients_1/add_5_1/add_grad/Reshape*)
_class
loc:@res1c_branch2b/BiasAdd*
data_formatNHWC*
T0
Ņ
-gradients_1/res1c_branch2b/MatMul_grad/MatMulMatMul$gradients_1/add_5_1/add_grad/Reshaperes1c_branch2b/kernel/read*
transpose_a( *
transpose_b(*
T0*(
_class
loc:@res1c_branch2b/MatMul
Ė
/gradients_1/res1c_branch2b/MatMul_grad/MatMul_1MatMulactivation_9/Relu$gradients_1/add_5_1/add_grad/Reshape*(
_class
loc:@res1c_branch2b/MatMul*
transpose_a(*
transpose_b( *
T0
Ø
+gradients_1/activation_9/Relu_grad/ReluGradReluGrad-gradients_1/res1c_branch2b/MatMul_grad/MatMulactivation_9/Relu*
T0*$
_class
loc:@activation_9/Relu
ŗ
3gradients_1/res1c_branch2a/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients_1/activation_9/Relu_grad/ReluGrad*
T0*)
_class
loc:@res1c_branch2a/BiasAdd*
data_formatNHWC
Ł
-gradients_1/res1c_branch2a/MatMul_grad/MatMulMatMul+gradients_1/activation_9/Relu_grad/ReluGradres1c_branch2a/kernel/read*
transpose_a( *
transpose_b(*
T0*(
_class
loc:@res1c_branch2a/MatMul
Ņ
/gradients_1/res1c_branch2a/MatMul_grad/MatMul_1MatMulactivation_8/Relu+gradients_1/activation_9/Relu_grad/ReluGrad*
T0*(
_class
loc:@res1c_branch2a/MatMul*
transpose_a(*
transpose_b( 
£
gradients_1/AddN_2AddN&gradients_1/add_5_1/add_grad/Reshape_1-gradients_1/res1c_branch2a/MatMul_grad/MatMul*
N*
T0*
_class
loc:@add_5_1/add

+gradients_1/activation_8/Relu_grad/ReluGradReluGradgradients_1/AddN_2activation_8/Relu*
T0*$
_class
loc:@activation_8/Relu
~
"gradients_1/add_4_1/add_grad/ShapeShaperes1b_branch2b_1/BiasAdd*
out_type0*
_class
loc:@add_4_1/add*
T0
y
$gradients_1/add_4_1/add_grad/Shape_1Shapeactivation_6/Relu*
T0*
out_type0*
_class
loc:@add_4_1/add
¾
2gradients_1/add_4_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_1/add_4_1/add_grad/Shape$gradients_1/add_4_1/add_grad/Shape_1*
T0*
_class
loc:@add_4_1/add
Ī
 gradients_1/add_4_1/add_grad/SumSum+gradients_1/activation_8/Relu_grad/ReluGrad2gradients_1/add_4_1/add_grad/BroadcastGradientArgs*
_class
loc:@add_4_1/add*

Tidx0*
	keep_dims( *
T0
¬
$gradients_1/add_4_1/add_grad/ReshapeReshape gradients_1/add_4_1/add_grad/Sum"gradients_1/add_4_1/add_grad/Shape*
T0*
Tshape0*
_class
loc:@add_4_1/add
Ņ
"gradients_1/add_4_1/add_grad/Sum_1Sum+gradients_1/activation_8/Relu_grad/ReluGrad4gradients_1/add_4_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_class
loc:@add_4_1/add
²
&gradients_1/add_4_1/add_grad/Reshape_1Reshape"gradients_1/add_4_1/add_grad/Sum_1$gradients_1/add_4_1/add_grad/Shape_1*
T0*
Tshape0*
_class
loc:@add_4_1/add
·
5gradients_1/res1b_branch2b_1/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients_1/add_4_1/add_grad/Reshape*
T0*+
_class!
loc:@res1b_branch2b_1/BiasAdd*
data_formatNHWC
Ų
/gradients_1/res1b_branch2b_1/MatMul_grad/MatMulMatMul$gradients_1/add_4_1/add_grad/Reshaperes1b_branch2b_1/kernel/read*
T0**
_class 
loc:@res1b_branch2b_1/MatMul*
transpose_a( *
transpose_b(
Ļ
1gradients_1/res1b_branch2b_1/MatMul_grad/MatMul_1MatMulactivation_7/Relu$gradients_1/add_4_1/add_grad/Reshape*
T0**
_class 
loc:@res1b_branch2b_1/MatMul*
transpose_a(*
transpose_b( 
Ŗ
+gradients_1/activation_7/Relu_grad/ReluGradReluGrad/gradients_1/res1b_branch2b_1/MatMul_grad/MatMulactivation_7/Relu*$
_class
loc:@activation_7/Relu*
T0
¾
5gradients_1/res1b_branch2a_1/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients_1/activation_7/Relu_grad/ReluGrad*+
_class!
loc:@res1b_branch2a_1/BiasAdd*
data_formatNHWC*
T0
ß
/gradients_1/res1b_branch2a_1/MatMul_grad/MatMulMatMul+gradients_1/activation_7/Relu_grad/ReluGradres1b_branch2a_1/kernel/read*
T0**
_class 
loc:@res1b_branch2a_1/MatMul*
transpose_a( *
transpose_b(
Ö
1gradients_1/res1b_branch2a_1/MatMul_grad/MatMul_1MatMulactivation_6/Relu+gradients_1/activation_7/Relu_grad/ReluGrad*
T0**
_class 
loc:@res1b_branch2a_1/MatMul*
transpose_a(*
transpose_b( 
„
gradients_1/AddN_3AddN&gradients_1/add_4_1/add_grad/Reshape_1/gradients_1/res1b_branch2a_1/MatMul_grad/MatMul*
N*
T0*
_class
loc:@add_4_1/add

+gradients_1/activation_6/Relu_grad/ReluGradReluGradgradients_1/AddN_3activation_6/Relu*$
_class
loc:@activation_6/Relu*
T0
~
"gradients_1/add_3_1/add_grad/ShapeShaperes1a_branch2b_1/BiasAdd*
T0*
out_type0*
_class
loc:@add_3_1/add
t
$gradients_1/add_3_1/add_grad/Shape_1Shapedense_3/Relu*
T0*
out_type0*
_class
loc:@add_3_1/add
¾
2gradients_1/add_3_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_1/add_3_1/add_grad/Shape$gradients_1/add_3_1/add_grad/Shape_1*
T0*
_class
loc:@add_3_1/add
Ī
 gradients_1/add_3_1/add_grad/SumSum+gradients_1/activation_6/Relu_grad/ReluGrad2gradients_1/add_3_1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_class
loc:@add_3_1/add
¬
$gradients_1/add_3_1/add_grad/ReshapeReshape gradients_1/add_3_1/add_grad/Sum"gradients_1/add_3_1/add_grad/Shape*
T0*
Tshape0*
_class
loc:@add_3_1/add
Ņ
"gradients_1/add_3_1/add_grad/Sum_1Sum+gradients_1/activation_6/Relu_grad/ReluGrad4gradients_1/add_3_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_class
loc:@add_3_1/add
²
&gradients_1/add_3_1/add_grad/Reshape_1Reshape"gradients_1/add_3_1/add_grad/Sum_1$gradients_1/add_3_1/add_grad/Shape_1*
T0*
Tshape0*
_class
loc:@add_3_1/add
·
5gradients_1/res1a_branch2b_1/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients_1/add_3_1/add_grad/Reshape*
data_formatNHWC*
T0*+
_class!
loc:@res1a_branch2b_1/BiasAdd
Ų
/gradients_1/res1a_branch2b_1/MatMul_grad/MatMulMatMul$gradients_1/add_3_1/add_grad/Reshaperes1a_branch2b_1/kernel/read*
transpose_b(*
T0**
_class 
loc:@res1a_branch2b_1/MatMul*
transpose_a( 
Ļ
1gradients_1/res1a_branch2b_1/MatMul_grad/MatMul_1MatMulactivation_5/Relu$gradients_1/add_3_1/add_grad/Reshape*
transpose_b( *
T0**
_class 
loc:@res1a_branch2b_1/MatMul*
transpose_a(
Ŗ
+gradients_1/activation_5/Relu_grad/ReluGradReluGrad/gradients_1/res1a_branch2b_1/MatMul_grad/MatMulactivation_5/Relu*
T0*$
_class
loc:@activation_5/Relu
¾
5gradients_1/res1a_branch2a_1/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients_1/activation_5/Relu_grad/ReluGrad*
T0*+
_class!
loc:@res1a_branch2a_1/BiasAdd*
data_formatNHWC
ß
/gradients_1/res1a_branch2a_1/MatMul_grad/MatMulMatMul+gradients_1/activation_5/Relu_grad/ReluGradres1a_branch2a_1/kernel/read*
T0**
_class 
loc:@res1a_branch2a_1/MatMul*
transpose_a( *
transpose_b(
Ń
1gradients_1/res1a_branch2a_1/MatMul_grad/MatMul_1MatMuldense_3/Relu+gradients_1/activation_5/Relu_grad/ReluGrad*
transpose_b( *
T0**
_class 
loc:@res1a_branch2a_1/MatMul*
transpose_a(
„
gradients_1/AddN_4AddN&gradients_1/add_3_1/add_grad/Reshape_1/gradients_1/res1a_branch2a_1/MatMul_grad/MatMul*
T0*
_class
loc:@add_3_1/add*
N
~
&gradients_1/dense_3/Relu_grad/ReluGradReluGradgradients_1/AddN_4dense_3/Relu*
T0*
_class
loc:@dense_3/Relu
§
,gradients_1/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_1/dense_3/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_3/BiasAdd*
data_formatNHWC
æ
&gradients_1/dense_3/MatMul_grad/MatMulMatMul&gradients_1/dense_3/Relu_grad/ReluGraddense_3/kernel/read*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a( *
transpose_b(
µ
(gradients_1/dense_3/MatMul_grad/MatMul_1MatMulinput_2&gradients_1/dense_3/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a(*
transpose_b( 
>
AssignAdd_1/valueConst*
valueB
 *  ?*
dtype0
v
AssignAdd_1	AssignAdditerations_1AssignAdd_1/value*
use_locking( *
T0*
_class
loc:@iterations_1
5
add_39/yConst*
dtype0*
valueB
 *  ?
3
add_39Additerations_1/readadd_39/y*
T0
,
Pow_2Powbeta_2_1/readadd_39*
T0
5
sub_40/xConst*
valueB
 *  ?*
dtype0
'
sub_40Subsub_40/xPow_2*
T0
5
Const_56Const*
valueB
 *    *
dtype0
5
Const_57Const*
valueB
 *  *
dtype0
>
clip_by_value_13/MinimumMinimumsub_40Const_57*
T0
H
clip_by_value_13Maximumclip_by_value_13/MinimumConst_56*
T0
*
Sqrt_13Sqrtclip_by_value_13*
T0
,
Pow_3Powbeta_1_1/readadd_39*
T0
5
sub_41/xConst*
valueB
 *  ?*
dtype0
'
sub_41Subsub_41/xPow_3*
T0
/

truediv_15RealDivSqrt_13sub_41*
T0
-
mul_65Mul	lr_1/read
truediv_15*
T0
:
Const_58Const*
valueBŲ*    *
dtype0
\
Variable_24
VariableV2*
shared_name *
dtype0*
	container *
shape:Ų

Variable_24/AssignAssignVariable_24Const_58*
use_locking(*
T0*
_class
loc:@Variable_24*
validate_shape(
R
Variable_24/readIdentityVariable_24*
_class
loc:@Variable_24*
T0
>
Const_59Const*
valueB	Ų*    *
dtype0
`
Variable_25
VariableV2*
dtype0*
	container *
shape:	Ų*
shared_name 

Variable_25/AssignAssignVariable_25Const_59*
_class
loc:@Variable_25*
validate_shape(*
use_locking(*
T0
R
Variable_25/readIdentityVariable_25*
_class
loc:@Variable_25*
T0
9
Const_60Const*
valueB*    *
dtype0
[
Variable_26
VariableV2*
dtype0*
	container *
shape:*
shared_name 

Variable_26/AssignAssignVariable_26Const_60*
T0*
_class
loc:@Variable_26*
validate_shape(*
use_locking(
R
Variable_26/readIdentityVariable_26*
T0*
_class
loc:@Variable_26
>
Const_61Const*
valueB	Ų*    *
dtype0
`
Variable_27
VariableV2*
dtype0*
	container *
shape:	Ų*
shared_name 

Variable_27/AssignAssignVariable_27Const_61*
T0*
_class
loc:@Variable_27*
validate_shape(*
use_locking(
R
Variable_27/readIdentityVariable_27*
T0*
_class
loc:@Variable_27
:
Const_62Const*
valueBŲ*    *
dtype0
\
Variable_28
VariableV2*
shared_name *
dtype0*
	container *
shape:Ų

Variable_28/AssignAssignVariable_28Const_62*
use_locking(*
T0*
_class
loc:@Variable_28*
validate_shape(
R
Variable_28/readIdentityVariable_28*
T0*
_class
loc:@Variable_28
?
Const_63Const*
valueB
ŲŲ*    *
dtype0
a
Variable_29
VariableV2*
shape:
ŲŲ*
shared_name *
dtype0*
	container 

Variable_29/AssignAssignVariable_29Const_63*
use_locking(*
T0*
_class
loc:@Variable_29*
validate_shape(
R
Variable_29/readIdentityVariable_29*
T0*
_class
loc:@Variable_29
:
Const_64Const*
dtype0*
valueBŲ*    
\
Variable_30
VariableV2*
	container *
shape:Ų*
shared_name *
dtype0

Variable_30/AssignAssignVariable_30Const_64*
_class
loc:@Variable_30*
validate_shape(*
use_locking(*
T0
R
Variable_30/readIdentityVariable_30*
T0*
_class
loc:@Variable_30
?
Const_65Const*
valueB
ŲŲ*    *
dtype0
a
Variable_31
VariableV2*
shape:
ŲŲ*
shared_name *
dtype0*
	container 

Variable_31/AssignAssignVariable_31Const_65*
T0*
_class
loc:@Variable_31*
validate_shape(*
use_locking(
R
Variable_31/readIdentityVariable_31*
T0*
_class
loc:@Variable_31
:
Const_66Const*
valueBŲ*    *
dtype0
\
Variable_32
VariableV2*
shape:Ų*
shared_name *
dtype0*
	container 

Variable_32/AssignAssignVariable_32Const_66*
use_locking(*
T0*
_class
loc:@Variable_32*
validate_shape(
R
Variable_32/readIdentityVariable_32*
T0*
_class
loc:@Variable_32
?
Const_67Const*
valueB
ŲŲ*    *
dtype0
a
Variable_33
VariableV2*
shape:
ŲŲ*
shared_name *
dtype0*
	container 

Variable_33/AssignAssignVariable_33Const_67*
_class
loc:@Variable_33*
validate_shape(*
use_locking(*
T0
R
Variable_33/readIdentityVariable_33*
T0*
_class
loc:@Variable_33
:
Const_68Const*
valueBŲ*    *
dtype0
\
Variable_34
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 

Variable_34/AssignAssignVariable_34Const_68*
use_locking(*
T0*
_class
loc:@Variable_34*
validate_shape(
R
Variable_34/readIdentityVariable_34*
_class
loc:@Variable_34*
T0
?
Const_69Const*
valueB
ŲŲ*    *
dtype0
a
Variable_35
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 

Variable_35/AssignAssignVariable_35Const_69*
use_locking(*
T0*
_class
loc:@Variable_35*
validate_shape(
R
Variable_35/readIdentityVariable_35*
T0*
_class
loc:@Variable_35
:
Const_70Const*
valueBŲ*    *
dtype0
\
Variable_36
VariableV2*
shape:Ų*
shared_name *
dtype0*
	container 

Variable_36/AssignAssignVariable_36Const_70*
_class
loc:@Variable_36*
validate_shape(*
use_locking(*
T0
R
Variable_36/readIdentityVariable_36*
T0*
_class
loc:@Variable_36
?
Const_71Const*
valueB
ŲŲ*    *
dtype0
a
Variable_37
VariableV2*
shared_name *
dtype0*
	container *
shape:
ŲŲ

Variable_37/AssignAssignVariable_37Const_71*
T0*
_class
loc:@Variable_37*
validate_shape(*
use_locking(
R
Variable_37/readIdentityVariable_37*
_class
loc:@Variable_37*
T0
:
Const_72Const*
valueBŲ*    *
dtype0
\
Variable_38
VariableV2*
	container *
shape:Ų*
shared_name *
dtype0

Variable_38/AssignAssignVariable_38Const_72*
use_locking(*
T0*
_class
loc:@Variable_38*
validate_shape(
R
Variable_38/readIdentityVariable_38*
T0*
_class
loc:@Variable_38
?
Const_73Const*
valueB
ŲŲ*    *
dtype0
a
Variable_39
VariableV2*
shared_name *
dtype0*
	container *
shape:
ŲŲ

Variable_39/AssignAssignVariable_39Const_73*
T0*
_class
loc:@Variable_39*
validate_shape(*
use_locking(
R
Variable_39/readIdentityVariable_39*
T0*
_class
loc:@Variable_39
:
Const_74Const*
valueBŲ*    *
dtype0
\
Variable_40
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 

Variable_40/AssignAssignVariable_40Const_74*
_class
loc:@Variable_40*
validate_shape(*
use_locking(*
T0
R
Variable_40/readIdentityVariable_40*
T0*
_class
loc:@Variable_40
?
Const_75Const*
valueB
ŲŲ*    *
dtype0
a
Variable_41
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 

Variable_41/AssignAssignVariable_41Const_75*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_41
R
Variable_41/readIdentityVariable_41*
T0*
_class
loc:@Variable_41
:
Const_76Const*
valueBŲ*    *
dtype0
\
Variable_42
VariableV2*
shared_name *
dtype0*
	container *
shape:Ų

Variable_42/AssignAssignVariable_42Const_76*
_class
loc:@Variable_42*
validate_shape(*
use_locking(*
T0
R
Variable_42/readIdentityVariable_42*
T0*
_class
loc:@Variable_42
?
Const_77Const*
valueB
ŲŲ*    *
dtype0
a
Variable_43
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 

Variable_43/AssignAssignVariable_43Const_77*
use_locking(*
T0*
_class
loc:@Variable_43*
validate_shape(
R
Variable_43/readIdentityVariable_43*
T0*
_class
loc:@Variable_43
:
Const_78Const*
valueBŲ*    *
dtype0
\
Variable_44
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 

Variable_44/AssignAssignVariable_44Const_78*
use_locking(*
T0*
_class
loc:@Variable_44*
validate_shape(
R
Variable_44/readIdentityVariable_44*
T0*
_class
loc:@Variable_44
?
Const_79Const*
valueB
ŲŲ*    *
dtype0
a
Variable_45
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 

Variable_45/AssignAssignVariable_45Const_79*
use_locking(*
T0*
_class
loc:@Variable_45*
validate_shape(
R
Variable_45/readIdentityVariable_45*
T0*
_class
loc:@Variable_45
:
Const_80Const*
valueBŲ*    *
dtype0
\
Variable_46
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 

Variable_46/AssignAssignVariable_46Const_80*
use_locking(*
T0*
_class
loc:@Variable_46*
validate_shape(
R
Variable_46/readIdentityVariable_46*
T0*
_class
loc:@Variable_46
?
Const_81Const*
valueB
ŲŲ*    *
dtype0
a
Variable_47
VariableV2*
shared_name *
dtype0*
	container *
shape:
ŲŲ

Variable_47/AssignAssignVariable_47Const_81*
use_locking(*
T0*
_class
loc:@Variable_47*
validate_shape(
R
Variable_47/readIdentityVariable_47*
T0*
_class
loc:@Variable_47
:
Const_82Const*
valueBŲ*    *
dtype0
\
Variable_48
VariableV2*
shared_name *
dtype0*
	container *
shape:Ų

Variable_48/AssignAssignVariable_48Const_82*
use_locking(*
T0*
_class
loc:@Variable_48*
validate_shape(
R
Variable_48/readIdentityVariable_48*
_class
loc:@Variable_48*
T0
>
Const_83Const*
valueB	Ų*    *
dtype0
`
Variable_49
VariableV2*
shape:	Ų*
shared_name *
dtype0*
	container 

Variable_49/AssignAssignVariable_49Const_83*
use_locking(*
T0*
_class
loc:@Variable_49*
validate_shape(
R
Variable_49/readIdentityVariable_49*
T0*
_class
loc:@Variable_49
9
Const_84Const*
valueB*    *
dtype0
[
Variable_50
VariableV2*
	container *
shape:*
shared_name *
dtype0

Variable_50/AssignAssignVariable_50Const_84*
_class
loc:@Variable_50*
validate_shape(*
use_locking(*
T0
R
Variable_50/readIdentityVariable_50*
T0*
_class
loc:@Variable_50
>
Const_85Const*
valueB	Ų*    *
dtype0
`
Variable_51
VariableV2*
dtype0*
	container *
shape:	Ų*
shared_name 

Variable_51/AssignAssignVariable_51Const_85*
use_locking(*
T0*
_class
loc:@Variable_51*
validate_shape(
R
Variable_51/readIdentityVariable_51*
T0*
_class
loc:@Variable_51
:
Const_86Const*
valueBŲ*    *
dtype0
\
Variable_52
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 

Variable_52/AssignAssignVariable_52Const_86*
use_locking(*
T0*
_class
loc:@Variable_52*
validate_shape(
R
Variable_52/readIdentityVariable_52*
T0*
_class
loc:@Variable_52
?
Const_87Const*
valueB
ŲŲ*    *
dtype0
a
Variable_53
VariableV2*
shared_name *
dtype0*
	container *
shape:
ŲŲ

Variable_53/AssignAssignVariable_53Const_87*
use_locking(*
T0*
_class
loc:@Variable_53*
validate_shape(
R
Variable_53/readIdentityVariable_53*
T0*
_class
loc:@Variable_53
:
Const_88Const*
valueBŲ*    *
dtype0
\
Variable_54
VariableV2*
shared_name *
dtype0*
	container *
shape:Ų

Variable_54/AssignAssignVariable_54Const_88*
use_locking(*
T0*
_class
loc:@Variable_54*
validate_shape(
R
Variable_54/readIdentityVariable_54*
T0*
_class
loc:@Variable_54
?
Const_89Const*
valueB
ŲŲ*    *
dtype0
a
Variable_55
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 

Variable_55/AssignAssignVariable_55Const_89*
use_locking(*
T0*
_class
loc:@Variable_55*
validate_shape(
R
Variable_55/readIdentityVariable_55*
_class
loc:@Variable_55*
T0
:
Const_90Const*
valueBŲ*    *
dtype0
\
Variable_56
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 

Variable_56/AssignAssignVariable_56Const_90*
use_locking(*
T0*
_class
loc:@Variable_56*
validate_shape(
R
Variable_56/readIdentityVariable_56*
T0*
_class
loc:@Variable_56
?
Const_91Const*
valueB
ŲŲ*    *
dtype0
a
Variable_57
VariableV2*
shared_name *
dtype0*
	container *
shape:
ŲŲ

Variable_57/AssignAssignVariable_57Const_91*
_class
loc:@Variable_57*
validate_shape(*
use_locking(*
T0
R
Variable_57/readIdentityVariable_57*
T0*
_class
loc:@Variable_57
:
Const_92Const*
valueBŲ*    *
dtype0
\
Variable_58
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 

Variable_58/AssignAssignVariable_58Const_92*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_58
R
Variable_58/readIdentityVariable_58*
T0*
_class
loc:@Variable_58
?
Const_93Const*
valueB
ŲŲ*    *
dtype0
a
Variable_59
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 

Variable_59/AssignAssignVariable_59Const_93*
use_locking(*
T0*
_class
loc:@Variable_59*
validate_shape(
R
Variable_59/readIdentityVariable_59*
_class
loc:@Variable_59*
T0
:
Const_94Const*
valueBŲ*    *
dtype0
\
Variable_60
VariableV2*
	container *
shape:Ų*
shared_name *
dtype0

Variable_60/AssignAssignVariable_60Const_94*
use_locking(*
T0*
_class
loc:@Variable_60*
validate_shape(
R
Variable_60/readIdentityVariable_60*
T0*
_class
loc:@Variable_60
?
Const_95Const*
valueB
ŲŲ*    *
dtype0
a
Variable_61
VariableV2*
	container *
shape:
ŲŲ*
shared_name *
dtype0

Variable_61/AssignAssignVariable_61Const_95*
_class
loc:@Variable_61*
validate_shape(*
use_locking(*
T0
R
Variable_61/readIdentityVariable_61*
_class
loc:@Variable_61*
T0
:
Const_96Const*
valueBŲ*    *
dtype0
\
Variable_62
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 

Variable_62/AssignAssignVariable_62Const_96*
use_locking(*
T0*
_class
loc:@Variable_62*
validate_shape(
R
Variable_62/readIdentityVariable_62*
T0*
_class
loc:@Variable_62
?
Const_97Const*
valueB
ŲŲ*    *
dtype0
a
Variable_63
VariableV2*
	container *
shape:
ŲŲ*
shared_name *
dtype0

Variable_63/AssignAssignVariable_63Const_97*
use_locking(*
T0*
_class
loc:@Variable_63*
validate_shape(
R
Variable_63/readIdentityVariable_63*
_class
loc:@Variable_63*
T0
:
Const_98Const*
valueBŲ*    *
dtype0
\
Variable_64
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 

Variable_64/AssignAssignVariable_64Const_98*
T0*
_class
loc:@Variable_64*
validate_shape(*
use_locking(
R
Variable_64/readIdentityVariable_64*
T0*
_class
loc:@Variable_64
?
Const_99Const*
valueB
ŲŲ*    *
dtype0
a
Variable_65
VariableV2*
shape:
ŲŲ*
shared_name *
dtype0*
	container 

Variable_65/AssignAssignVariable_65Const_99*
use_locking(*
T0*
_class
loc:@Variable_65*
validate_shape(
R
Variable_65/readIdentityVariable_65*
T0*
_class
loc:@Variable_65
;
	Const_100Const*
valueBŲ*    *
dtype0
\
Variable_66
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 

Variable_66/AssignAssignVariable_66	Const_100*
use_locking(*
T0*
_class
loc:@Variable_66*
validate_shape(
R
Variable_66/readIdentityVariable_66*
T0*
_class
loc:@Variable_66
@
	Const_101Const*
valueB
ŲŲ*    *
dtype0
a
Variable_67
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 

Variable_67/AssignAssignVariable_67	Const_101*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_67
R
Variable_67/readIdentityVariable_67*
T0*
_class
loc:@Variable_67
;
	Const_102Const*
valueBŲ*    *
dtype0
\
Variable_68
VariableV2*
shared_name *
dtype0*
	container *
shape:Ų

Variable_68/AssignAssignVariable_68	Const_102*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_68
R
Variable_68/readIdentityVariable_68*
T0*
_class
loc:@Variable_68
@
	Const_103Const*
valueB
ŲŲ*    *
dtype0
a
Variable_69
VariableV2*
dtype0*
	container *
shape:
ŲŲ*
shared_name 

Variable_69/AssignAssignVariable_69	Const_103*
use_locking(*
T0*
_class
loc:@Variable_69*
validate_shape(
R
Variable_69/readIdentityVariable_69*
T0*
_class
loc:@Variable_69
;
	Const_104Const*
valueBŲ*    *
dtype0
\
Variable_70
VariableV2*
dtype0*
	container *
shape:Ų*
shared_name 

Variable_70/AssignAssignVariable_70	Const_104*
use_locking(*
T0*
_class
loc:@Variable_70*
validate_shape(
R
Variable_70/readIdentityVariable_70*
_class
loc:@Variable_70*
T0
@
	Const_105Const*
valueB
ŲŲ*    *
dtype0
a
Variable_71
VariableV2*
shape:
ŲŲ*
shared_name *
dtype0*
	container 

Variable_71/AssignAssignVariable_71	Const_105*
use_locking(*
T0*
_class
loc:@Variable_71*
validate_shape(
R
Variable_71/readIdentityVariable_71*
_class
loc:@Variable_71*
T0
7
mul_66Mulbeta_1_1/readVariable_24/read*
T0
5
sub_42/xConst*
valueB
 *  ?*
dtype0
/
sub_42Subsub_42/xbeta_1_1/read*
T0
L
mul_67Mulsub_42,gradients_1/dense_3/BiasAdd_grad/BiasAddGrad*
T0
&
add_40Addmul_66mul_67*
T0
7
mul_68Mulbeta_2_1/readVariable_48/read*
T0
5
sub_43/xConst*
valueB
 *  ?*
dtype0
/
sub_43Subsub_43/xbeta_2_1/read*
T0
J
	Square_14Square,gradients_1/dense_3/BiasAdd_grad/BiasAddGrad*
T0
)
mul_69Mulsub_43	Square_14*
T0
&
add_41Addmul_68mul_69*
T0
&
mul_70Mulmul_65add_40*
T0
6
	Const_106Const*
valueB
 *    *
dtype0
6
	Const_107Const*
valueB
 *  *
dtype0
?
clip_by_value_14/MinimumMinimumadd_41	Const_107*
T0
I
clip_by_value_14Maximumclip_by_value_14/Minimum	Const_106*
T0
*
Sqrt_14Sqrtclip_by_value_14*
T0
5
add_42/yConst*
valueB
 *wĢ+2*
dtype0
)
add_42AddSqrt_14add_42/y*
T0
.

truediv_16RealDivmul_70add_42*
T0
5
sub_44Subdense_3/bias/read
truediv_16*
T0
z
	Assign_48AssignVariable_24add_40*
T0*
_class
loc:@Variable_24*
validate_shape(*
use_locking(
z
	Assign_49AssignVariable_48add_41*
use_locking(*
T0*
_class
loc:@Variable_48*
validate_shape(
|
	Assign_50Assigndense_3/biassub_44*
validate_shape(*
use_locking(*
T0*
_class
loc:@dense_3/bias
7
mul_71Mulbeta_1_1/readVariable_25/read*
T0
5
sub_45/xConst*
valueB
 *  ?*
dtype0
/
sub_45Subsub_45/xbeta_1_1/read*
T0
H
mul_72Mulsub_45(gradients_1/dense_3/MatMul_grad/MatMul_1*
T0
&
add_43Addmul_71mul_72*
T0
7
mul_73Mulbeta_2_1/readVariable_49/read*
T0
5
sub_46/xConst*
valueB
 *  ?*
dtype0
/
sub_46Subsub_46/xbeta_2_1/read*
T0
F
	Square_15Square(gradients_1/dense_3/MatMul_grad/MatMul_1*
T0
)
mul_74Mulsub_46	Square_15*
T0
&
add_44Addmul_73mul_74*
T0
&
mul_75Mulmul_65add_43*
T0
6
	Const_108Const*
valueB
 *    *
dtype0
6
	Const_109Const*
valueB
 *  *
dtype0
?
clip_by_value_15/MinimumMinimumadd_44	Const_109*
T0
I
clip_by_value_15Maximumclip_by_value_15/Minimum	Const_108*
T0
*
Sqrt_15Sqrtclip_by_value_15*
T0
5
add_45/yConst*
valueB
 *wĢ+2*
dtype0
)
add_45AddSqrt_15add_45/y*
T0
.

truediv_17RealDivmul_75add_45*
T0
7
sub_47Subdense_3/kernel/read
truediv_17*
T0
z
	Assign_51AssignVariable_25add_43*
use_locking(*
T0*
_class
loc:@Variable_25*
validate_shape(
z
	Assign_52AssignVariable_49add_44*
use_locking(*
T0*
_class
loc:@Variable_49*
validate_shape(

	Assign_53Assigndense_3/kernelsub_47*!
_class
loc:@dense_3/kernel*
validate_shape(*
use_locking(*
T0
7
mul_76Mulbeta_1_1/readVariable_26/read*
T0
5
sub_48/xConst*
valueB
 *  ?*
dtype0
/
sub_48Subsub_48/xbeta_1_1/read*
T0
L
mul_77Mulsub_48,gradients_1/dense_4/BiasAdd_grad/BiasAddGrad*
T0
&
add_46Addmul_76mul_77*
T0
7
mul_78Mulbeta_2_1/readVariable_50/read*
T0
5
sub_49/xConst*
valueB
 *  ?*
dtype0
/
sub_49Subsub_49/xbeta_2_1/read*
T0
J
	Square_16Square,gradients_1/dense_4/BiasAdd_grad/BiasAddGrad*
T0
)
mul_79Mulsub_49	Square_16*
T0
&
add_47Addmul_78mul_79*
T0
&
mul_80Mulmul_65add_46*
T0
6
	Const_110Const*
valueB
 *    *
dtype0
6
	Const_111Const*
valueB
 *  *
dtype0
?
clip_by_value_16/MinimumMinimumadd_47	Const_111*
T0
I
clip_by_value_16Maximumclip_by_value_16/Minimum	Const_110*
T0
*
Sqrt_16Sqrtclip_by_value_16*
T0
5
add_48/yConst*
valueB
 *wĢ+2*
dtype0
)
add_48AddSqrt_16add_48/y*
T0
.

truediv_18RealDivmul_80add_48*
T0
5
sub_50Subdense_4/bias/read
truediv_18*
T0
z
	Assign_54AssignVariable_26add_46*
_class
loc:@Variable_26*
validate_shape(*
use_locking(*
T0
z
	Assign_55AssignVariable_50add_47*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_50
|
	Assign_56Assigndense_4/biassub_50*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(
7
mul_81Mulbeta_1_1/readVariable_27/read*
T0
5
sub_51/xConst*
valueB
 *  ?*
dtype0
/
sub_51Subsub_51/xbeta_1_1/read*
T0
H
mul_82Mulsub_51(gradients_1/dense_4/MatMul_grad/MatMul_1*
T0
&
add_49Addmul_81mul_82*
T0
7
mul_83Mulbeta_2_1/readVariable_51/read*
T0
5
sub_52/xConst*
valueB
 *  ?*
dtype0
/
sub_52Subsub_52/xbeta_2_1/read*
T0
F
	Square_17Square(gradients_1/dense_4/MatMul_grad/MatMul_1*
T0
)
mul_84Mulsub_52	Square_17*
T0
&
add_50Addmul_83mul_84*
T0
&
mul_85Mulmul_65add_49*
T0
6
	Const_112Const*
valueB
 *    *
dtype0
6
	Const_113Const*
valueB
 *  *
dtype0
?
clip_by_value_17/MinimumMinimumadd_50	Const_113*
T0
I
clip_by_value_17Maximumclip_by_value_17/Minimum	Const_112*
T0
*
Sqrt_17Sqrtclip_by_value_17*
T0
5
add_51/yConst*
valueB
 *wĢ+2*
dtype0
)
add_51AddSqrt_17add_51/y*
T0
.

truediv_19RealDivmul_85add_51*
T0
7
sub_53Subdense_4/kernel/read
truediv_19*
T0
z
	Assign_57AssignVariable_27add_49*
T0*
_class
loc:@Variable_27*
validate_shape(*
use_locking(
z
	Assign_58AssignVariable_51add_50*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_51

	Assign_59Assigndense_4/kernelsub_53*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(
7
mul_86Mulbeta_1_1/readVariable_28/read*
T0
5
sub_54/xConst*
dtype0*
valueB
 *  ?
/
sub_54Subsub_54/xbeta_1_1/read*
T0
U
mul_87Mulsub_545gradients_1/res1a_branch2a_1/BiasAdd_grad/BiasAddGrad*
T0
&
add_52Addmul_86mul_87*
T0
7
mul_88Mulbeta_2_1/readVariable_52/read*
T0
5
sub_55/xConst*
valueB
 *  ?*
dtype0
/
sub_55Subsub_55/xbeta_2_1/read*
T0
S
	Square_18Square5gradients_1/res1a_branch2a_1/BiasAdd_grad/BiasAddGrad*
T0
)
mul_89Mulsub_55	Square_18*
T0
&
add_53Addmul_88mul_89*
T0
&
mul_90Mulmul_65add_52*
T0
6
	Const_114Const*
valueB
 *    *
dtype0
6
	Const_115Const*
valueB
 *  *
dtype0
?
clip_by_value_18/MinimumMinimumadd_53	Const_115*
T0
I
clip_by_value_18Maximumclip_by_value_18/Minimum	Const_114*
T0
*
Sqrt_18Sqrtclip_by_value_18*
T0
5
add_54/yConst*
valueB
 *wĢ+2*
dtype0
)
add_54AddSqrt_18add_54/y*
T0
.

truediv_20RealDivmul_90add_54*
T0
>
sub_56Subres1a_branch2a_1/bias/read
truediv_20*
T0
z
	Assign_60AssignVariable_28add_52*
T0*
_class
loc:@Variable_28*
validate_shape(*
use_locking(
z
	Assign_61AssignVariable_52add_53*
use_locking(*
T0*
_class
loc:@Variable_52*
validate_shape(

	Assign_62Assignres1a_branch2a_1/biassub_56*
validate_shape(*
use_locking(*
T0*(
_class
loc:@res1a_branch2a_1/bias
7
mul_91Mulbeta_1_1/readVariable_29/read*
T0
5
sub_57/xConst*
valueB
 *  ?*
dtype0
/
sub_57Subsub_57/xbeta_1_1/read*
T0
Q
mul_92Mulsub_571gradients_1/res1a_branch2a_1/MatMul_grad/MatMul_1*
T0
&
add_55Addmul_91mul_92*
T0
7
mul_93Mulbeta_2_1/readVariable_53/read*
T0
5
sub_58/xConst*
valueB
 *  ?*
dtype0
/
sub_58Subsub_58/xbeta_2_1/read*
T0
O
	Square_19Square1gradients_1/res1a_branch2a_1/MatMul_grad/MatMul_1*
T0
)
mul_94Mulsub_58	Square_19*
T0
&
add_56Addmul_93mul_94*
T0
&
mul_95Mulmul_65add_55*
T0
6
	Const_116Const*
valueB
 *    *
dtype0
6
	Const_117Const*
valueB
 *  *
dtype0
?
clip_by_value_19/MinimumMinimumadd_56	Const_117*
T0
I
clip_by_value_19Maximumclip_by_value_19/Minimum	Const_116*
T0
*
Sqrt_19Sqrtclip_by_value_19*
T0
5
add_57/yConst*
valueB
 *wĢ+2*
dtype0
)
add_57AddSqrt_19add_57/y*
T0
.

truediv_21RealDivmul_95add_57*
T0
@
sub_59Subres1a_branch2a_1/kernel/read
truediv_21*
T0
z
	Assign_63AssignVariable_29add_55*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_29
z
	Assign_64AssignVariable_53add_56*
use_locking(*
T0*
_class
loc:@Variable_53*
validate_shape(

	Assign_65Assignres1a_branch2a_1/kernelsub_59*
validate_shape(*
use_locking(*
T0**
_class 
loc:@res1a_branch2a_1/kernel
7
mul_96Mulbeta_1_1/readVariable_30/read*
T0
5
sub_60/xConst*
valueB
 *  ?*
dtype0
/
sub_60Subsub_60/xbeta_1_1/read*
T0
U
mul_97Mulsub_605gradients_1/res1a_branch2b_1/BiasAdd_grad/BiasAddGrad*
T0
&
add_58Addmul_96mul_97*
T0
7
mul_98Mulbeta_2_1/readVariable_54/read*
T0
5
sub_61/xConst*
valueB
 *  ?*
dtype0
/
sub_61Subsub_61/xbeta_2_1/read*
T0
S
	Square_20Square5gradients_1/res1a_branch2b_1/BiasAdd_grad/BiasAddGrad*
T0
)
mul_99Mulsub_61	Square_20*
T0
&
add_59Addmul_98mul_99*
T0
'
mul_100Mulmul_65add_58*
T0
6
	Const_118Const*
valueB
 *    *
dtype0
6
	Const_119Const*
valueB
 *  *
dtype0
?
clip_by_value_20/MinimumMinimumadd_59	Const_119*
T0
I
clip_by_value_20Maximumclip_by_value_20/Minimum	Const_118*
T0
*
Sqrt_20Sqrtclip_by_value_20*
T0
5
add_60/yConst*
valueB
 *wĢ+2*
dtype0
)
add_60AddSqrt_20add_60/y*
T0
/

truediv_22RealDivmul_100add_60*
T0
>
sub_62Subres1a_branch2b_1/bias/read
truediv_22*
T0
z
	Assign_66AssignVariable_30add_58*
T0*
_class
loc:@Variable_30*
validate_shape(*
use_locking(
z
	Assign_67AssignVariable_54add_59*
use_locking(*
T0*
_class
loc:@Variable_54*
validate_shape(

	Assign_68Assignres1a_branch2b_1/biassub_62*
validate_shape(*
use_locking(*
T0*(
_class
loc:@res1a_branch2b_1/bias
8
mul_101Mulbeta_1_1/readVariable_31/read*
T0
5
sub_63/xConst*
valueB
 *  ?*
dtype0
/
sub_63Subsub_63/xbeta_1_1/read*
T0
R
mul_102Mulsub_631gradients_1/res1a_branch2b_1/MatMul_grad/MatMul_1*
T0
(
add_61Addmul_101mul_102*
T0
8
mul_103Mulbeta_2_1/readVariable_55/read*
T0
5
sub_64/xConst*
valueB
 *  ?*
dtype0
/
sub_64Subsub_64/xbeta_2_1/read*
T0
O
	Square_21Square1gradients_1/res1a_branch2b_1/MatMul_grad/MatMul_1*
T0
*
mul_104Mulsub_64	Square_21*
T0
(
add_62Addmul_103mul_104*
T0
'
mul_105Mulmul_65add_61*
T0
6
	Const_120Const*
valueB
 *    *
dtype0
6
	Const_121Const*
valueB
 *  *
dtype0
?
clip_by_value_21/MinimumMinimumadd_62	Const_121*
T0
I
clip_by_value_21Maximumclip_by_value_21/Minimum	Const_120*
T0
*
Sqrt_21Sqrtclip_by_value_21*
T0
5
add_63/yConst*
valueB
 *wĢ+2*
dtype0
)
add_63AddSqrt_21add_63/y*
T0
/

truediv_23RealDivmul_105add_63*
T0
@
sub_65Subres1a_branch2b_1/kernel/read
truediv_23*
T0
z
	Assign_69AssignVariable_31add_61*
use_locking(*
T0*
_class
loc:@Variable_31*
validate_shape(
z
	Assign_70AssignVariable_55add_62*
use_locking(*
T0*
_class
loc:@Variable_55*
validate_shape(

	Assign_71Assignres1a_branch2b_1/kernelsub_65*
use_locking(*
T0**
_class 
loc:@res1a_branch2b_1/kernel*
validate_shape(
8
mul_106Mulbeta_1_1/readVariable_32/read*
T0
5
sub_66/xConst*
valueB
 *  ?*
dtype0
/
sub_66Subsub_66/xbeta_1_1/read*
T0
V
mul_107Mulsub_665gradients_1/res1b_branch2a_1/BiasAdd_grad/BiasAddGrad*
T0
(
add_64Addmul_106mul_107*
T0
8
mul_108Mulbeta_2_1/readVariable_56/read*
T0
5
sub_67/xConst*
valueB
 *  ?*
dtype0
/
sub_67Subsub_67/xbeta_2_1/read*
T0
S
	Square_22Square5gradients_1/res1b_branch2a_1/BiasAdd_grad/BiasAddGrad*
T0
*
mul_109Mulsub_67	Square_22*
T0
(
add_65Addmul_108mul_109*
T0
'
mul_110Mulmul_65add_64*
T0
6
	Const_122Const*
valueB
 *    *
dtype0
6
	Const_123Const*
valueB
 *  *
dtype0
?
clip_by_value_22/MinimumMinimumadd_65	Const_123*
T0
I
clip_by_value_22Maximumclip_by_value_22/Minimum	Const_122*
T0
*
Sqrt_22Sqrtclip_by_value_22*
T0
5
add_66/yConst*
valueB
 *wĢ+2*
dtype0
)
add_66AddSqrt_22add_66/y*
T0
/

truediv_24RealDivmul_110add_66*
T0
>
sub_68Subres1b_branch2a_1/bias/read
truediv_24*
T0
z
	Assign_72AssignVariable_32add_64*
use_locking(*
T0*
_class
loc:@Variable_32*
validate_shape(
z
	Assign_73AssignVariable_56add_65*
T0*
_class
loc:@Variable_56*
validate_shape(*
use_locking(

	Assign_74Assignres1b_branch2a_1/biassub_68*
use_locking(*
T0*(
_class
loc:@res1b_branch2a_1/bias*
validate_shape(
8
mul_111Mulbeta_1_1/readVariable_33/read*
T0
5
sub_69/xConst*
valueB
 *  ?*
dtype0
/
sub_69Subsub_69/xbeta_1_1/read*
T0
R
mul_112Mulsub_691gradients_1/res1b_branch2a_1/MatMul_grad/MatMul_1*
T0
(
add_67Addmul_111mul_112*
T0
8
mul_113Mulbeta_2_1/readVariable_57/read*
T0
5
sub_70/xConst*
valueB
 *  ?*
dtype0
/
sub_70Subsub_70/xbeta_2_1/read*
T0
O
	Square_23Square1gradients_1/res1b_branch2a_1/MatMul_grad/MatMul_1*
T0
*
mul_114Mulsub_70	Square_23*
T0
(
add_68Addmul_113mul_114*
T0
'
mul_115Mulmul_65add_67*
T0
6
	Const_124Const*
valueB
 *    *
dtype0
6
	Const_125Const*
valueB
 *  *
dtype0
?
clip_by_value_23/MinimumMinimumadd_68	Const_125*
T0
I
clip_by_value_23Maximumclip_by_value_23/Minimum	Const_124*
T0
*
Sqrt_23Sqrtclip_by_value_23*
T0
5
add_69/yConst*
valueB
 *wĢ+2*
dtype0
)
add_69AddSqrt_23add_69/y*
T0
/

truediv_25RealDivmul_115add_69*
T0
@
sub_71Subres1b_branch2a_1/kernel/read
truediv_25*
T0
z
	Assign_75AssignVariable_33add_67*
use_locking(*
T0*
_class
loc:@Variable_33*
validate_shape(
z
	Assign_76AssignVariable_57add_68*
_class
loc:@Variable_57*
validate_shape(*
use_locking(*
T0

	Assign_77Assignres1b_branch2a_1/kernelsub_71*
use_locking(*
T0**
_class 
loc:@res1b_branch2a_1/kernel*
validate_shape(
8
mul_116Mulbeta_1_1/readVariable_34/read*
T0
5
sub_72/xConst*
valueB
 *  ?*
dtype0
/
sub_72Subsub_72/xbeta_1_1/read*
T0
V
mul_117Mulsub_725gradients_1/res1b_branch2b_1/BiasAdd_grad/BiasAddGrad*
T0
(
add_70Addmul_116mul_117*
T0
8
mul_118Mulbeta_2_1/readVariable_58/read*
T0
5
sub_73/xConst*
valueB
 *  ?*
dtype0
/
sub_73Subsub_73/xbeta_2_1/read*
T0
S
	Square_24Square5gradients_1/res1b_branch2b_1/BiasAdd_grad/BiasAddGrad*
T0
*
mul_119Mulsub_73	Square_24*
T0
(
add_71Addmul_118mul_119*
T0
'
mul_120Mulmul_65add_70*
T0
6
	Const_126Const*
valueB
 *    *
dtype0
6
	Const_127Const*
valueB
 *  *
dtype0
?
clip_by_value_24/MinimumMinimumadd_71	Const_127*
T0
I
clip_by_value_24Maximumclip_by_value_24/Minimum	Const_126*
T0
*
Sqrt_24Sqrtclip_by_value_24*
T0
5
add_72/yConst*
valueB
 *wĢ+2*
dtype0
)
add_72AddSqrt_24add_72/y*
T0
/

truediv_26RealDivmul_120add_72*
T0
>
sub_74Subres1b_branch2b_1/bias/read
truediv_26*
T0
z
	Assign_78AssignVariable_34add_70*
use_locking(*
T0*
_class
loc:@Variable_34*
validate_shape(
z
	Assign_79AssignVariable_58add_71*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_58

	Assign_80Assignres1b_branch2b_1/biassub_74*
use_locking(*
T0*(
_class
loc:@res1b_branch2b_1/bias*
validate_shape(
8
mul_121Mulbeta_1_1/readVariable_35/read*
T0
5
sub_75/xConst*
valueB
 *  ?*
dtype0
/
sub_75Subsub_75/xbeta_1_1/read*
T0
R
mul_122Mulsub_751gradients_1/res1b_branch2b_1/MatMul_grad/MatMul_1*
T0
(
add_73Addmul_121mul_122*
T0
8
mul_123Mulbeta_2_1/readVariable_59/read*
T0
5
sub_76/xConst*
valueB
 *  ?*
dtype0
/
sub_76Subsub_76/xbeta_2_1/read*
T0
O
	Square_25Square1gradients_1/res1b_branch2b_1/MatMul_grad/MatMul_1*
T0
*
mul_124Mulsub_76	Square_25*
T0
(
add_74Addmul_123mul_124*
T0
'
mul_125Mulmul_65add_73*
T0
6
	Const_128Const*
dtype0*
valueB
 *    
6
	Const_129Const*
valueB
 *  *
dtype0
?
clip_by_value_25/MinimumMinimumadd_74	Const_129*
T0
I
clip_by_value_25Maximumclip_by_value_25/Minimum	Const_128*
T0
*
Sqrt_25Sqrtclip_by_value_25*
T0
5
add_75/yConst*
valueB
 *wĢ+2*
dtype0
)
add_75AddSqrt_25add_75/y*
T0
/

truediv_27RealDivmul_125add_75*
T0
@
sub_77Subres1b_branch2b_1/kernel/read
truediv_27*
T0
z
	Assign_81AssignVariable_35add_73*
use_locking(*
T0*
_class
loc:@Variable_35*
validate_shape(
z
	Assign_82AssignVariable_59add_74*
use_locking(*
T0*
_class
loc:@Variable_59*
validate_shape(

	Assign_83Assignres1b_branch2b_1/kernelsub_77*
use_locking(*
T0**
_class 
loc:@res1b_branch2b_1/kernel*
validate_shape(
8
mul_126Mulbeta_1_1/readVariable_36/read*
T0
5
sub_78/xConst*
valueB
 *  ?*
dtype0
/
sub_78Subsub_78/xbeta_1_1/read*
T0
T
mul_127Mulsub_783gradients_1/res1c_branch2a/BiasAdd_grad/BiasAddGrad*
T0
(
add_76Addmul_126mul_127*
T0
8
mul_128Mulbeta_2_1/readVariable_60/read*
T0
5
sub_79/xConst*
valueB
 *  ?*
dtype0
/
sub_79Subsub_79/xbeta_2_1/read*
T0
Q
	Square_26Square3gradients_1/res1c_branch2a/BiasAdd_grad/BiasAddGrad*
T0
*
mul_129Mulsub_79	Square_26*
T0
(
add_77Addmul_128mul_129*
T0
'
mul_130Mulmul_65add_76*
T0
6
	Const_130Const*
valueB
 *    *
dtype0
6
	Const_131Const*
valueB
 *  *
dtype0
?
clip_by_value_26/MinimumMinimumadd_77	Const_131*
T0
I
clip_by_value_26Maximumclip_by_value_26/Minimum	Const_130*
T0
*
Sqrt_26Sqrtclip_by_value_26*
T0
5
add_78/yConst*
valueB
 *wĢ+2*
dtype0
)
add_78AddSqrt_26add_78/y*
T0
/

truediv_28RealDivmul_130add_78*
T0
<
sub_80Subres1c_branch2a/bias/read
truediv_28*
T0
z
	Assign_84AssignVariable_36add_76*
use_locking(*
T0*
_class
loc:@Variable_36*
validate_shape(
z
	Assign_85AssignVariable_60add_77*
use_locking(*
T0*
_class
loc:@Variable_60*
validate_shape(

	Assign_86Assignres1c_branch2a/biassub_80*
use_locking(*
T0*&
_class
loc:@res1c_branch2a/bias*
validate_shape(
8
mul_131Mulbeta_1_1/readVariable_37/read*
T0
5
sub_81/xConst*
valueB
 *  ?*
dtype0
/
sub_81Subsub_81/xbeta_1_1/read*
T0
P
mul_132Mulsub_81/gradients_1/res1c_branch2a/MatMul_grad/MatMul_1*
T0
(
add_79Addmul_131mul_132*
T0
8
mul_133Mulbeta_2_1/readVariable_61/read*
T0
5
sub_82/xConst*
dtype0*
valueB
 *  ?
/
sub_82Subsub_82/xbeta_2_1/read*
T0
M
	Square_27Square/gradients_1/res1c_branch2a/MatMul_grad/MatMul_1*
T0
*
mul_134Mulsub_82	Square_27*
T0
(
add_80Addmul_133mul_134*
T0
'
mul_135Mulmul_65add_79*
T0
6
	Const_132Const*
valueB
 *    *
dtype0
6
	Const_133Const*
valueB
 *  *
dtype0
?
clip_by_value_27/MinimumMinimumadd_80	Const_133*
T0
I
clip_by_value_27Maximumclip_by_value_27/Minimum	Const_132*
T0
*
Sqrt_27Sqrtclip_by_value_27*
T0
5
add_81/yConst*
valueB
 *wĢ+2*
dtype0
)
add_81AddSqrt_27add_81/y*
T0
/

truediv_29RealDivmul_135add_81*
T0
>
sub_83Subres1c_branch2a/kernel/read
truediv_29*
T0
z
	Assign_87AssignVariable_37add_79*
use_locking(*
T0*
_class
loc:@Variable_37*
validate_shape(
z
	Assign_88AssignVariable_61add_80*
use_locking(*
T0*
_class
loc:@Variable_61*
validate_shape(

	Assign_89Assignres1c_branch2a/kernelsub_83*
use_locking(*
T0*(
_class
loc:@res1c_branch2a/kernel*
validate_shape(
8
mul_136Mulbeta_1_1/readVariable_38/read*
T0
5
sub_84/xConst*
valueB
 *  ?*
dtype0
/
sub_84Subsub_84/xbeta_1_1/read*
T0
T
mul_137Mulsub_843gradients_1/res1c_branch2b/BiasAdd_grad/BiasAddGrad*
T0
(
add_82Addmul_136mul_137*
T0
8
mul_138Mulbeta_2_1/readVariable_62/read*
T0
5
sub_85/xConst*
valueB
 *  ?*
dtype0
/
sub_85Subsub_85/xbeta_2_1/read*
T0
Q
	Square_28Square3gradients_1/res1c_branch2b/BiasAdd_grad/BiasAddGrad*
T0
*
mul_139Mulsub_85	Square_28*
T0
(
add_83Addmul_138mul_139*
T0
'
mul_140Mulmul_65add_82*
T0
6
	Const_134Const*
valueB
 *    *
dtype0
6
	Const_135Const*
valueB
 *  *
dtype0
?
clip_by_value_28/MinimumMinimumadd_83	Const_135*
T0
I
clip_by_value_28Maximumclip_by_value_28/Minimum	Const_134*
T0
*
Sqrt_28Sqrtclip_by_value_28*
T0
5
add_84/yConst*
valueB
 *wĢ+2*
dtype0
)
add_84AddSqrt_28add_84/y*
T0
/

truediv_30RealDivmul_140add_84*
T0
<
sub_86Subres1c_branch2b/bias/read
truediv_30*
T0
z
	Assign_90AssignVariable_38add_82*
_class
loc:@Variable_38*
validate_shape(*
use_locking(*
T0
z
	Assign_91AssignVariable_62add_83*
use_locking(*
T0*
_class
loc:@Variable_62*
validate_shape(

	Assign_92Assignres1c_branch2b/biassub_86*
use_locking(*
T0*&
_class
loc:@res1c_branch2b/bias*
validate_shape(
8
mul_141Mulbeta_1_1/readVariable_39/read*
T0
5
sub_87/xConst*
dtype0*
valueB
 *  ?
/
sub_87Subsub_87/xbeta_1_1/read*
T0
P
mul_142Mulsub_87/gradients_1/res1c_branch2b/MatMul_grad/MatMul_1*
T0
(
add_85Addmul_141mul_142*
T0
8
mul_143Mulbeta_2_1/readVariable_63/read*
T0
5
sub_88/xConst*
valueB
 *  ?*
dtype0
/
sub_88Subsub_88/xbeta_2_1/read*
T0
M
	Square_29Square/gradients_1/res1c_branch2b/MatMul_grad/MatMul_1*
T0
*
mul_144Mulsub_88	Square_29*
T0
(
add_86Addmul_143mul_144*
T0
'
mul_145Mulmul_65add_85*
T0
6
	Const_136Const*
valueB
 *    *
dtype0
6
	Const_137Const*
valueB
 *  *
dtype0
?
clip_by_value_29/MinimumMinimumadd_86	Const_137*
T0
I
clip_by_value_29Maximumclip_by_value_29/Minimum	Const_136*
T0
*
Sqrt_29Sqrtclip_by_value_29*
T0
5
add_87/yConst*
valueB
 *wĢ+2*
dtype0
)
add_87AddSqrt_29add_87/y*
T0
/

truediv_31RealDivmul_145add_87*
T0
>
sub_89Subres1c_branch2b/kernel/read
truediv_31*
T0
z
	Assign_93AssignVariable_39add_85*
T0*
_class
loc:@Variable_39*
validate_shape(*
use_locking(
z
	Assign_94AssignVariable_63add_86*
use_locking(*
T0*
_class
loc:@Variable_63*
validate_shape(

	Assign_95Assignres1c_branch2b/kernelsub_89*
use_locking(*
T0*(
_class
loc:@res1c_branch2b/kernel*
validate_shape(
8
mul_146Mulbeta_1_1/readVariable_40/read*
T0
5
sub_90/xConst*
valueB
 *  ?*
dtype0
/
sub_90Subsub_90/xbeta_1_1/read*
T0
T
mul_147Mulsub_903gradients_1/res1d_branch2a/BiasAdd_grad/BiasAddGrad*
T0
(
add_88Addmul_146mul_147*
T0
8
mul_148Mulbeta_2_1/readVariable_64/read*
T0
5
sub_91/xConst*
valueB
 *  ?*
dtype0
/
sub_91Subsub_91/xbeta_2_1/read*
T0
Q
	Square_30Square3gradients_1/res1d_branch2a/BiasAdd_grad/BiasAddGrad*
T0
*
mul_149Mulsub_91	Square_30*
T0
(
add_89Addmul_148mul_149*
T0
'
mul_150Mulmul_65add_88*
T0
6
	Const_138Const*
valueB
 *    *
dtype0
6
	Const_139Const*
valueB
 *  *
dtype0
?
clip_by_value_30/MinimumMinimumadd_89	Const_139*
T0
I
clip_by_value_30Maximumclip_by_value_30/Minimum	Const_138*
T0
*
Sqrt_30Sqrtclip_by_value_30*
T0
5
add_90/yConst*
valueB
 *wĢ+2*
dtype0
)
add_90AddSqrt_30add_90/y*
T0
/

truediv_32RealDivmul_150add_90*
T0
<
sub_92Subres1d_branch2a/bias/read
truediv_32*
T0
z
	Assign_96AssignVariable_40add_88*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_40
z
	Assign_97AssignVariable_64add_89*
T0*
_class
loc:@Variable_64*
validate_shape(*
use_locking(

	Assign_98Assignres1d_branch2a/biassub_92*
use_locking(*
T0*&
_class
loc:@res1d_branch2a/bias*
validate_shape(
8
mul_151Mulbeta_1_1/readVariable_41/read*
T0
5
sub_93/xConst*
dtype0*
valueB
 *  ?
/
sub_93Subsub_93/xbeta_1_1/read*
T0
P
mul_152Mulsub_93/gradients_1/res1d_branch2a/MatMul_grad/MatMul_1*
T0
(
add_91Addmul_151mul_152*
T0
8
mul_153Mulbeta_2_1/readVariable_65/read*
T0
5
sub_94/xConst*
valueB
 *  ?*
dtype0
/
sub_94Subsub_94/xbeta_2_1/read*
T0
M
	Square_31Square/gradients_1/res1d_branch2a/MatMul_grad/MatMul_1*
T0
*
mul_154Mulsub_94	Square_31*
T0
(
add_92Addmul_153mul_154*
T0
'
mul_155Mulmul_65add_91*
T0
6
	Const_140Const*
valueB
 *    *
dtype0
6
	Const_141Const*
dtype0*
valueB
 *  
?
clip_by_value_31/MinimumMinimumadd_92	Const_141*
T0
I
clip_by_value_31Maximumclip_by_value_31/Minimum	Const_140*
T0
*
Sqrt_31Sqrtclip_by_value_31*
T0
5
add_93/yConst*
valueB
 *wĢ+2*
dtype0
)
add_93AddSqrt_31add_93/y*
T0
/

truediv_33RealDivmul_155add_93*
T0
>
sub_95Subres1d_branch2a/kernel/read
truediv_33*
T0
z
	Assign_99AssignVariable_41add_91*
use_locking(*
T0*
_class
loc:@Variable_41*
validate_shape(
{

Assign_100AssignVariable_65add_92*
_class
loc:@Variable_65*
validate_shape(*
use_locking(*
T0


Assign_101Assignres1d_branch2a/kernelsub_95*
use_locking(*
T0*(
_class
loc:@res1d_branch2a/kernel*
validate_shape(
8
mul_156Mulbeta_1_1/readVariable_42/read*
T0
5
sub_96/xConst*
valueB
 *  ?*
dtype0
/
sub_96Subsub_96/xbeta_1_1/read*
T0
T
mul_157Mulsub_963gradients_1/res1d_branch2b/BiasAdd_grad/BiasAddGrad*
T0
(
add_94Addmul_156mul_157*
T0
8
mul_158Mulbeta_2_1/readVariable_66/read*
T0
5
sub_97/xConst*
valueB
 *  ?*
dtype0
/
sub_97Subsub_97/xbeta_2_1/read*
T0
Q
	Square_32Square3gradients_1/res1d_branch2b/BiasAdd_grad/BiasAddGrad*
T0
*
mul_159Mulsub_97	Square_32*
T0
(
add_95Addmul_158mul_159*
T0
'
mul_160Mulmul_65add_94*
T0
6
	Const_142Const*
valueB
 *    *
dtype0
6
	Const_143Const*
valueB
 *  *
dtype0
?
clip_by_value_32/MinimumMinimumadd_95	Const_143*
T0
I
clip_by_value_32Maximumclip_by_value_32/Minimum	Const_142*
T0
*
Sqrt_32Sqrtclip_by_value_32*
T0
5
add_96/yConst*
valueB
 *wĢ+2*
dtype0
)
add_96AddSqrt_32add_96/y*
T0
/

truediv_34RealDivmul_160add_96*
T0
<
sub_98Subres1d_branch2b/bias/read
truediv_34*
T0
{

Assign_102AssignVariable_42add_94*
_class
loc:@Variable_42*
validate_shape(*
use_locking(*
T0
{

Assign_103AssignVariable_66add_95*
use_locking(*
T0*
_class
loc:@Variable_66*
validate_shape(


Assign_104Assignres1d_branch2b/biassub_98*
validate_shape(*
use_locking(*
T0*&
_class
loc:@res1d_branch2b/bias
8
mul_161Mulbeta_1_1/readVariable_43/read*
T0
5
sub_99/xConst*
valueB
 *  ?*
dtype0
/
sub_99Subsub_99/xbeta_1_1/read*
T0
P
mul_162Mulsub_99/gradients_1/res1d_branch2b/MatMul_grad/MatMul_1*
T0
(
add_97Addmul_161mul_162*
T0
8
mul_163Mulbeta_2_1/readVariable_67/read*
T0
6
	sub_100/xConst*
valueB
 *  ?*
dtype0
1
sub_100Sub	sub_100/xbeta_2_1/read*
T0
M
	Square_33Square/gradients_1/res1d_branch2b/MatMul_grad/MatMul_1*
T0
+
mul_164Mulsub_100	Square_33*
T0
(
add_98Addmul_163mul_164*
T0
'
mul_165Mulmul_65add_97*
T0
6
	Const_144Const*
valueB
 *    *
dtype0
6
	Const_145Const*
valueB
 *  *
dtype0
?
clip_by_value_33/MinimumMinimumadd_98	Const_145*
T0
I
clip_by_value_33Maximumclip_by_value_33/Minimum	Const_144*
T0
*
Sqrt_33Sqrtclip_by_value_33*
T0
5
add_99/yConst*
valueB
 *wĢ+2*
dtype0
)
add_99AddSqrt_33add_99/y*
T0
/

truediv_35RealDivmul_165add_99*
T0
?
sub_101Subres1d_branch2b/kernel/read
truediv_35*
T0
{

Assign_105AssignVariable_43add_97*
use_locking(*
T0*
_class
loc:@Variable_43*
validate_shape(
{

Assign_106AssignVariable_67add_98*
use_locking(*
T0*
_class
loc:@Variable_67*
validate_shape(


Assign_107Assignres1d_branch2b/kernelsub_101*
use_locking(*
T0*(
_class
loc:@res1d_branch2b/kernel*
validate_shape(
8
mul_166Mulbeta_1_1/readVariable_44/read*
T0
6
	sub_102/xConst*
valueB
 *  ?*
dtype0
1
sub_102Sub	sub_102/xbeta_1_1/read*
T0
U
mul_167Mulsub_1023gradients_1/res1e_branch2a/BiasAdd_grad/BiasAddGrad*
T0
)
add_100Addmul_166mul_167*
T0
8
mul_168Mulbeta_2_1/readVariable_68/read*
T0
6
	sub_103/xConst*
valueB
 *  ?*
dtype0
1
sub_103Sub	sub_103/xbeta_2_1/read*
T0
Q
	Square_34Square3gradients_1/res1e_branch2a/BiasAdd_grad/BiasAddGrad*
T0
+
mul_169Mulsub_103	Square_34*
T0
)
add_101Addmul_168mul_169*
T0
(
mul_170Mulmul_65add_100*
T0
6
	Const_146Const*
valueB
 *    *
dtype0
6
	Const_147Const*
dtype0*
valueB
 *  
@
clip_by_value_34/MinimumMinimumadd_101	Const_147*
T0
I
clip_by_value_34Maximumclip_by_value_34/Minimum	Const_146*
T0
*
Sqrt_34Sqrtclip_by_value_34*
T0
6
	add_102/yConst*
valueB
 *wĢ+2*
dtype0
+
add_102AddSqrt_34	add_102/y*
T0
0

truediv_36RealDivmul_170add_102*
T0
=
sub_104Subres1e_branch2a/bias/read
truediv_36*
T0
|

Assign_108AssignVariable_44add_100*
use_locking(*
T0*
_class
loc:@Variable_44*
validate_shape(
|

Assign_109AssignVariable_68add_101*
use_locking(*
T0*
_class
loc:@Variable_68*
validate_shape(


Assign_110Assignres1e_branch2a/biassub_104*
use_locking(*
T0*&
_class
loc:@res1e_branch2a/bias*
validate_shape(
8
mul_171Mulbeta_1_1/readVariable_45/read*
T0
6
	sub_105/xConst*
valueB
 *  ?*
dtype0
1
sub_105Sub	sub_105/xbeta_1_1/read*
T0
Q
mul_172Mulsub_105/gradients_1/res1e_branch2a/MatMul_grad/MatMul_1*
T0
)
add_103Addmul_171mul_172*
T0
8
mul_173Mulbeta_2_1/readVariable_69/read*
T0
6
	sub_106/xConst*
valueB
 *  ?*
dtype0
1
sub_106Sub	sub_106/xbeta_2_1/read*
T0
M
	Square_35Square/gradients_1/res1e_branch2a/MatMul_grad/MatMul_1*
T0
+
mul_174Mulsub_106	Square_35*
T0
)
add_104Addmul_173mul_174*
T0
(
mul_175Mulmul_65add_103*
T0
6
	Const_148Const*
valueB
 *    *
dtype0
6
	Const_149Const*
valueB
 *  *
dtype0
@
clip_by_value_35/MinimumMinimumadd_104	Const_149*
T0
I
clip_by_value_35Maximumclip_by_value_35/Minimum	Const_148*
T0
*
Sqrt_35Sqrtclip_by_value_35*
T0
6
	add_105/yConst*
valueB
 *wĢ+2*
dtype0
+
add_105AddSqrt_35	add_105/y*
T0
0

truediv_37RealDivmul_175add_105*
T0
?
sub_107Subres1e_branch2a/kernel/read
truediv_37*
T0
|

Assign_111AssignVariable_45add_103*
T0*
_class
loc:@Variable_45*
validate_shape(*
use_locking(
|

Assign_112AssignVariable_69add_104*
_class
loc:@Variable_69*
validate_shape(*
use_locking(*
T0


Assign_113Assignres1e_branch2a/kernelsub_107*
use_locking(*
T0*(
_class
loc:@res1e_branch2a/kernel*
validate_shape(
8
mul_176Mulbeta_1_1/readVariable_46/read*
T0
6
	sub_108/xConst*
valueB
 *  ?*
dtype0
1
sub_108Sub	sub_108/xbeta_1_1/read*
T0
U
mul_177Mulsub_1083gradients_1/res1e_branch2b/BiasAdd_grad/BiasAddGrad*
T0
)
add_106Addmul_176mul_177*
T0
8
mul_178Mulbeta_2_1/readVariable_70/read*
T0
6
	sub_109/xConst*
valueB
 *  ?*
dtype0
1
sub_109Sub	sub_109/xbeta_2_1/read*
T0
Q
	Square_36Square3gradients_1/res1e_branch2b/BiasAdd_grad/BiasAddGrad*
T0
+
mul_179Mulsub_109	Square_36*
T0
)
add_107Addmul_178mul_179*
T0
(
mul_180Mulmul_65add_106*
T0
6
	Const_150Const*
valueB
 *    *
dtype0
6
	Const_151Const*
valueB
 *  *
dtype0
@
clip_by_value_36/MinimumMinimumadd_107	Const_151*
T0
I
clip_by_value_36Maximumclip_by_value_36/Minimum	Const_150*
T0
*
Sqrt_36Sqrtclip_by_value_36*
T0
6
	add_108/yConst*
valueB
 *wĢ+2*
dtype0
+
add_108AddSqrt_36	add_108/y*
T0
0

truediv_38RealDivmul_180add_108*
T0
=
sub_110Subres1e_branch2b/bias/read
truediv_38*
T0
|

Assign_114AssignVariable_46add_106*
use_locking(*
T0*
_class
loc:@Variable_46*
validate_shape(
|

Assign_115AssignVariable_70add_107*
_class
loc:@Variable_70*
validate_shape(*
use_locking(*
T0


Assign_116Assignres1e_branch2b/biassub_110*
use_locking(*
T0*&
_class
loc:@res1e_branch2b/bias*
validate_shape(
8
mul_181Mulbeta_1_1/readVariable_47/read*
T0
6
	sub_111/xConst*
valueB
 *  ?*
dtype0
1
sub_111Sub	sub_111/xbeta_1_1/read*
T0
Q
mul_182Mulsub_111/gradients_1/res1e_branch2b/MatMul_grad/MatMul_1*
T0
)
add_109Addmul_181mul_182*
T0
8
mul_183Mulbeta_2_1/readVariable_71/read*
T0
6
	sub_112/xConst*
valueB
 *  ?*
dtype0
1
sub_112Sub	sub_112/xbeta_2_1/read*
T0
M
	Square_37Square/gradients_1/res1e_branch2b/MatMul_grad/MatMul_1*
T0
+
mul_184Mulsub_112	Square_37*
T0
)
add_110Addmul_183mul_184*
T0
(
mul_185Mulmul_65add_109*
T0
6
	Const_152Const*
valueB
 *    *
dtype0
6
	Const_153Const*
valueB
 *  *
dtype0
@
clip_by_value_37/MinimumMinimumadd_110	Const_153*
T0
I
clip_by_value_37Maximumclip_by_value_37/Minimum	Const_152*
T0
*
Sqrt_37Sqrtclip_by_value_37*
T0
6
	add_111/yConst*
valueB
 *wĢ+2*
dtype0
+
add_111AddSqrt_37	add_111/y*
T0
0

truediv_39RealDivmul_185add_111*
T0
?
sub_113Subres1e_branch2b/kernel/read
truediv_39*
T0
|

Assign_117AssignVariable_47add_109*
use_locking(*
T0*
_class
loc:@Variable_47*
validate_shape(
|

Assign_118AssignVariable_71add_110*
use_locking(*
T0*
_class
loc:@Variable_71*
validate_shape(


Assign_119Assignres1e_branch2b/kernelsub_113*
validate_shape(*
use_locking(*
T0*(
_class
loc:@res1e_branch2b/kernel
Ø
group_deps_4NoOp^AssignAdd_1^Assign_100^Assign_101^Assign_102^Assign_103^Assign_104^Assign_105^Assign_106^Assign_107^Assign_108^Assign_109^Assign_110^Assign_111^Assign_112^Assign_113^Assign_114^Assign_115^Assign_116^Assign_117^Assign_118^Assign_119
^Assign_48
^Assign_49
^Assign_50
^Assign_51
^Assign_52
^Assign_53
^Assign_54
^Assign_55
^Assign_56
^Assign_57
^Assign_58
^Assign_59
^Assign_60
^Assign_61
^Assign_62
^Assign_63
^Assign_64
^Assign_65
^Assign_66
^Assign_67
^Assign_68
^Assign_69
^Assign_70
^Assign_71
^Assign_72
^Assign_73
^Assign_74
^Assign_75
^Assign_76
^Assign_77
^Assign_78
^Assign_79
^Assign_80
^Assign_81
^Assign_82
^Assign_83
^Assign_84
^Assign_85
^Assign_86
^Assign_87
^Assign_88
^Assign_89
^Assign_90
^Assign_91
^Assign_92
^Assign_93
^Assign_94
^Assign_95
^Assign_96
^Assign_97
^Assign_98
^Assign_99^Mean_9^mul_64

init_1NoOp^Variable_24/Assign^Variable_25/Assign^Variable_26/Assign^Variable_27/Assign^Variable_28/Assign^Variable_29/Assign^Variable_30/Assign^Variable_31/Assign^Variable_32/Assign^Variable_33/Assign^Variable_34/Assign^Variable_35/Assign^Variable_36/Assign^Variable_37/Assign^Variable_38/Assign^Variable_39/Assign^Variable_40/Assign^Variable_41/Assign^Variable_42/Assign^Variable_43/Assign^Variable_44/Assign^Variable_45/Assign^Variable_46/Assign^Variable_47/Assign^Variable_48/Assign^Variable_49/Assign^Variable_50/Assign^Variable_51/Assign^Variable_52/Assign^Variable_53/Assign^Variable_54/Assign^Variable_55/Assign^Variable_56/Assign^Variable_57/Assign^Variable_58/Assign^Variable_59/Assign^Variable_60/Assign^Variable_61/Assign^Variable_62/Assign^Variable_63/Assign^Variable_64/Assign^Variable_65/Assign^Variable_66/Assign^Variable_67/Assign^Variable_68/Assign^Variable_69/Assign^Variable_70/Assign^Variable_71/Assign^beta_1_1/Assign^beta_2_1/Assign^decay_1/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^iterations_1/Assign^lr_1/Assign^res1a_branch2a_1/bias/Assign^res1a_branch2a_1/kernel/Assign^res1a_branch2b_1/bias/Assign^res1a_branch2b_1/kernel/Assign^res1b_branch2a_1/bias/Assign^res1b_branch2a_1/kernel/Assign^res1b_branch2b_1/bias/Assign^res1b_branch2b_1/kernel/Assign^res1c_branch2a/bias/Assign^res1c_branch2a/kernel/Assign^res1c_branch2b/bias/Assign^res1c_branch2b/kernel/Assign^res1d_branch2a/bias/Assign^res1d_branch2a/kernel/Assign^res1d_branch2b/bias/Assign^res1d_branch2b/kernel/Assign^res1e_branch2a/bias/Assign^res1e_branch2a/kernel/Assign^res1e_branch2b/bias/Assign^res1e_branch2b/kernel/Assign
@
Placeholder_12Placeholder*
dtype0*
shape:	Ų


Assign_120Assigndense_3/kernelPlaceholder_12*
use_locking( *
T0*!
_class
loc:@dense_3/kernel*
validate_shape(
<
Placeholder_13Placeholder*
dtype0*
shape:Ų


Assign_121Assigndense_3/biasPlaceholder_13*
_class
loc:@dense_3/bias*
validate_shape(*
use_locking( *
T0
A
Placeholder_14Placeholder*
dtype0*
shape:
ŲŲ


Assign_122Assignres1a_branch2a_1/kernelPlaceholder_14**
_class 
loc:@res1a_branch2a_1/kernel*
validate_shape(*
use_locking( *
T0
<
Placeholder_15Placeholder*
dtype0*
shape:Ų


Assign_123Assignres1a_branch2a_1/biasPlaceholder_15*
T0*(
_class
loc:@res1a_branch2a_1/bias*
validate_shape(*
use_locking( 
A
Placeholder_16Placeholder*
dtype0*
shape:
ŲŲ


Assign_124Assignres1a_branch2b_1/kernelPlaceholder_16*
validate_shape(*
use_locking( *
T0**
_class 
loc:@res1a_branch2b_1/kernel
<
Placeholder_17Placeholder*
dtype0*
shape:Ų


Assign_125Assignres1a_branch2b_1/biasPlaceholder_17*(
_class
loc:@res1a_branch2b_1/bias*
validate_shape(*
use_locking( *
T0
A
Placeholder_18Placeholder*
dtype0*
shape:
ŲŲ


Assign_126Assignres1b_branch2a_1/kernelPlaceholder_18*
T0**
_class 
loc:@res1b_branch2a_1/kernel*
validate_shape(*
use_locking( 
<
Placeholder_19Placeholder*
dtype0*
shape:Ų


Assign_127Assignres1b_branch2a_1/biasPlaceholder_19*
T0*(
_class
loc:@res1b_branch2a_1/bias*
validate_shape(*
use_locking( 
A
Placeholder_20Placeholder*
shape:
ŲŲ*
dtype0


Assign_128Assignres1b_branch2b_1/kernelPlaceholder_20*
use_locking( *
T0**
_class 
loc:@res1b_branch2b_1/kernel*
validate_shape(
<
Placeholder_21Placeholder*
dtype0*
shape:Ų


Assign_129Assignres1b_branch2b_1/biasPlaceholder_21*
T0*(
_class
loc:@res1b_branch2b_1/bias*
validate_shape(*
use_locking( 
A
Placeholder_22Placeholder*
dtype0*
shape:
ŲŲ


Assign_130Assignres1c_branch2a/kernelPlaceholder_22*
use_locking( *
T0*(
_class
loc:@res1c_branch2a/kernel*
validate_shape(
<
Placeholder_23Placeholder*
shape:Ų*
dtype0


Assign_131Assignres1c_branch2a/biasPlaceholder_23*
use_locking( *
T0*&
_class
loc:@res1c_branch2a/bias*
validate_shape(
A
Placeholder_24Placeholder*
dtype0*
shape:
ŲŲ


Assign_132Assignres1c_branch2b/kernelPlaceholder_24*(
_class
loc:@res1c_branch2b/kernel*
validate_shape(*
use_locking( *
T0
<
Placeholder_25Placeholder*
dtype0*
shape:Ų


Assign_133Assignres1c_branch2b/biasPlaceholder_25*
use_locking( *
T0*&
_class
loc:@res1c_branch2b/bias*
validate_shape(
A
Placeholder_26Placeholder*
shape:
ŲŲ*
dtype0


Assign_134Assignres1d_branch2a/kernelPlaceholder_26*
validate_shape(*
use_locking( *
T0*(
_class
loc:@res1d_branch2a/kernel
<
Placeholder_27Placeholder*
shape:Ų*
dtype0


Assign_135Assignres1d_branch2a/biasPlaceholder_27*
use_locking( *
T0*&
_class
loc:@res1d_branch2a/bias*
validate_shape(
A
Placeholder_28Placeholder*
shape:
ŲŲ*
dtype0


Assign_136Assignres1d_branch2b/kernelPlaceholder_28*
use_locking( *
T0*(
_class
loc:@res1d_branch2b/kernel*
validate_shape(
<
Placeholder_29Placeholder*
shape:Ų*
dtype0


Assign_137Assignres1d_branch2b/biasPlaceholder_29*
T0*&
_class
loc:@res1d_branch2b/bias*
validate_shape(*
use_locking( 
A
Placeholder_30Placeholder*
dtype0*
shape:
ŲŲ


Assign_138Assignres1e_branch2a/kernelPlaceholder_30*
use_locking( *
T0*(
_class
loc:@res1e_branch2a/kernel*
validate_shape(
<
Placeholder_31Placeholder*
dtype0*
shape:Ų


Assign_139Assignres1e_branch2a/biasPlaceholder_31*
use_locking( *
T0*&
_class
loc:@res1e_branch2a/bias*
validate_shape(
A
Placeholder_32Placeholder*
shape:
ŲŲ*
dtype0


Assign_140Assignres1e_branch2b/kernelPlaceholder_32*(
_class
loc:@res1e_branch2b/kernel*
validate_shape(*
use_locking( *
T0
<
Placeholder_33Placeholder*
shape:Ų*
dtype0


Assign_141Assignres1e_branch2b/biasPlaceholder_33*
use_locking( *
T0*&
_class
loc:@res1e_branch2b/bias*
validate_shape(
@
Placeholder_34Placeholder*
shape:	Ų*
dtype0


Assign_142Assigndense_4/kernelPlaceholder_34*
use_locking( *
T0*!
_class
loc:@dense_4/kernel*
validate_shape(
;
Placeholder_35Placeholder*
dtype0*
shape:


Assign_143Assigndense_4/biasPlaceholder_35*
use_locking( *
T0*
_class
loc:@dense_4/bias*
validate_shape(
&
group_deps_5NoOp^dense_4/BiasAdd
:
save_1/ConstConst*
valueB Bmodel*
dtype0
¬
save_1/SaveV2/tensor_namesConst*ł
valueļBģvBVariableB
Variable_1BVariable_10BVariable_11BVariable_12BVariable_13BVariable_14BVariable_15BVariable_16BVariable_17BVariable_18BVariable_19B
Variable_2BVariable_20BVariable_21BVariable_22BVariable_23BVariable_24BVariable_25BVariable_26BVariable_27BVariable_28BVariable_29B
Variable_3BVariable_30BVariable_31BVariable_32BVariable_33BVariable_34BVariable_35BVariable_36BVariable_37BVariable_38BVariable_39B
Variable_4BVariable_40BVariable_41BVariable_42BVariable_43BVariable_44BVariable_45BVariable_46BVariable_47BVariable_48BVariable_49B
Variable_5BVariable_50BVariable_51BVariable_52BVariable_53BVariable_54BVariable_55BVariable_56BVariable_57BVariable_58BVariable_59B
Variable_6BVariable_60BVariable_61BVariable_62BVariable_63BVariable_64BVariable_65BVariable_66BVariable_67BVariable_68BVariable_69B
Variable_7BVariable_70BVariable_71B
Variable_8B
Variable_9Bbeta_1Bbeta_1_1Bbeta_2Bbeta_2_1BdecayBdecay_1Bdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBdense_4/biasBdense_4/kernelB
iterationsBiterations_1BlrBlr_1Bres1a_branch2a/biasBres1a_branch2a/kernelBres1a_branch2a_1/biasBres1a_branch2a_1/kernelBres1a_branch2b/biasBres1a_branch2b/kernelBres1a_branch2b_1/biasBres1a_branch2b_1/kernelBres1b_branch2a/biasBres1b_branch2a/kernelBres1b_branch2a_1/biasBres1b_branch2a_1/kernelBres1b_branch2b/biasBres1b_branch2b/kernelBres1b_branch2b_1/biasBres1b_branch2b_1/kernelBres1c_branch2a/biasBres1c_branch2a/kernelBres1c_branch2b/biasBres1c_branch2b/kernelBres1d_branch2a/biasBres1d_branch2a/kernelBres1d_branch2b/biasBres1d_branch2b/kernelBres1e_branch2a/biasBres1e_branch2a/kernelBres1e_branch2b/biasBres1e_branch2b/kernel*
dtype0
ø
save_1/SaveV2/shape_and_slicesConst*
value÷BōvB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ģ
save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesVariable
Variable_1Variable_10Variable_11Variable_12Variable_13Variable_14Variable_15Variable_16Variable_17Variable_18Variable_19
Variable_2Variable_20Variable_21Variable_22Variable_23Variable_24Variable_25Variable_26Variable_27Variable_28Variable_29
Variable_3Variable_30Variable_31Variable_32Variable_33Variable_34Variable_35Variable_36Variable_37Variable_38Variable_39
Variable_4Variable_40Variable_41Variable_42Variable_43Variable_44Variable_45Variable_46Variable_47Variable_48Variable_49
Variable_5Variable_50Variable_51Variable_52Variable_53Variable_54Variable_55Variable_56Variable_57Variable_58Variable_59
Variable_6Variable_60Variable_61Variable_62Variable_63Variable_64Variable_65Variable_66Variable_67Variable_68Variable_69
Variable_7Variable_70Variable_71
Variable_8
Variable_9beta_1beta_1_1beta_2beta_2_1decaydecay_1dense_1/biasdense_1/kerneldense_2/biasdense_2/kerneldense_3/biasdense_3/kerneldense_4/biasdense_4/kernel
iterationsiterations_1lrlr_1res1a_branch2a/biasres1a_branch2a/kernelres1a_branch2a_1/biasres1a_branch2a_1/kernelres1a_branch2b/biasres1a_branch2b/kernelres1a_branch2b_1/biasres1a_branch2b_1/kernelres1b_branch2a/biasres1b_branch2a/kernelres1b_branch2a_1/biasres1b_branch2a_1/kernelres1b_branch2b/biasres1b_branch2b/kernelres1b_branch2b_1/biasres1b_branch2b_1/kernelres1c_branch2a/biasres1c_branch2a/kernelres1c_branch2b/biasres1c_branch2b/kernelres1d_branch2a/biasres1d_branch2a/kernelres1d_branch2b/biasres1d_branch2b/kernelres1e_branch2a/biasres1e_branch2a/kernelres1e_branch2b/biasres1e_branch2b/kernel*
dtypesz
x2v
m
save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
T0*
_class
loc:@save_1/Const
¾
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*ł
valueļBģvBVariableB
Variable_1BVariable_10BVariable_11BVariable_12BVariable_13BVariable_14BVariable_15BVariable_16BVariable_17BVariable_18BVariable_19B
Variable_2BVariable_20BVariable_21BVariable_22BVariable_23BVariable_24BVariable_25BVariable_26BVariable_27BVariable_28BVariable_29B
Variable_3BVariable_30BVariable_31BVariable_32BVariable_33BVariable_34BVariable_35BVariable_36BVariable_37BVariable_38BVariable_39B
Variable_4BVariable_40BVariable_41BVariable_42BVariable_43BVariable_44BVariable_45BVariable_46BVariable_47BVariable_48BVariable_49B
Variable_5BVariable_50BVariable_51BVariable_52BVariable_53BVariable_54BVariable_55BVariable_56BVariable_57BVariable_58BVariable_59B
Variable_6BVariable_60BVariable_61BVariable_62BVariable_63BVariable_64BVariable_65BVariable_66BVariable_67BVariable_68BVariable_69B
Variable_7BVariable_70BVariable_71B
Variable_8B
Variable_9Bbeta_1Bbeta_1_1Bbeta_2Bbeta_2_1BdecayBdecay_1Bdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBdense_4/biasBdense_4/kernelB
iterationsBiterations_1BlrBlr_1Bres1a_branch2a/biasBres1a_branch2a/kernelBres1a_branch2a_1/biasBres1a_branch2a_1/kernelBres1a_branch2b/biasBres1a_branch2b/kernelBres1a_branch2b_1/biasBres1a_branch2b_1/kernelBres1b_branch2a/biasBres1b_branch2a/kernelBres1b_branch2a_1/biasBres1b_branch2a_1/kernelBres1b_branch2b/biasBres1b_branch2b/kernelBres1b_branch2b_1/biasBres1b_branch2b_1/kernelBres1c_branch2a/biasBres1c_branch2a/kernelBres1c_branch2b/biasBres1c_branch2b/kernelBres1d_branch2a/biasBres1d_branch2a/kernelBres1d_branch2b/biasBres1d_branch2b/kernelBres1e_branch2a/biasBres1e_branch2a/kernelBres1e_branch2b/biasBres1e_branch2b/kernel*
dtype0
Ź
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
value÷BōvB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*
dtypesz
x2v

save_1/AssignAssignVariablesave_1/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(

save_1/Assign_1Assign
Variable_1save_1/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(

save_1/Assign_2AssignVariable_10save_1/RestoreV2:2*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_10

save_1/Assign_3AssignVariable_11save_1/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(

save_1/Assign_4AssignVariable_12save_1/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(

save_1/Assign_5AssignVariable_13save_1/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(

save_1/Assign_6AssignVariable_14save_1/RestoreV2:6*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(

save_1/Assign_7AssignVariable_15save_1/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(

save_1/Assign_8AssignVariable_16save_1/RestoreV2:8*
_class
loc:@Variable_16*
validate_shape(*
use_locking(*
T0

save_1/Assign_9AssignVariable_17save_1/RestoreV2:9*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(

save_1/Assign_10AssignVariable_18save_1/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(

save_1/Assign_11AssignVariable_19save_1/RestoreV2:11*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_19

save_1/Assign_12Assign
Variable_2save_1/RestoreV2:12*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(

save_1/Assign_13AssignVariable_20save_1/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(

save_1/Assign_14AssignVariable_21save_1/RestoreV2:14*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(

save_1/Assign_15AssignVariable_22save_1/RestoreV2:15*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(

save_1/Assign_16AssignVariable_23save_1/RestoreV2:16*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_23

save_1/Assign_17AssignVariable_24save_1/RestoreV2:17*
use_locking(*
T0*
_class
loc:@Variable_24*
validate_shape(

save_1/Assign_18AssignVariable_25save_1/RestoreV2:18*
T0*
_class
loc:@Variable_25*
validate_shape(*
use_locking(

save_1/Assign_19AssignVariable_26save_1/RestoreV2:19*
_class
loc:@Variable_26*
validate_shape(*
use_locking(*
T0

save_1/Assign_20AssignVariable_27save_1/RestoreV2:20*
_class
loc:@Variable_27*
validate_shape(*
use_locking(*
T0

save_1/Assign_21AssignVariable_28save_1/RestoreV2:21*
use_locking(*
T0*
_class
loc:@Variable_28*
validate_shape(

save_1/Assign_22AssignVariable_29save_1/RestoreV2:22*
use_locking(*
T0*
_class
loc:@Variable_29*
validate_shape(

save_1/Assign_23Assign
Variable_3save_1/RestoreV2:23*
_class
loc:@Variable_3*
validate_shape(*
use_locking(*
T0

save_1/Assign_24AssignVariable_30save_1/RestoreV2:24*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_30

save_1/Assign_25AssignVariable_31save_1/RestoreV2:25*
_class
loc:@Variable_31*
validate_shape(*
use_locking(*
T0

save_1/Assign_26AssignVariable_32save_1/RestoreV2:26*
use_locking(*
T0*
_class
loc:@Variable_32*
validate_shape(

save_1/Assign_27AssignVariable_33save_1/RestoreV2:27*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_33

save_1/Assign_28AssignVariable_34save_1/RestoreV2:28*
use_locking(*
T0*
_class
loc:@Variable_34*
validate_shape(

save_1/Assign_29AssignVariable_35save_1/RestoreV2:29*
use_locking(*
T0*
_class
loc:@Variable_35*
validate_shape(

save_1/Assign_30AssignVariable_36save_1/RestoreV2:30*
use_locking(*
T0*
_class
loc:@Variable_36*
validate_shape(

save_1/Assign_31AssignVariable_37save_1/RestoreV2:31*
use_locking(*
T0*
_class
loc:@Variable_37*
validate_shape(

save_1/Assign_32AssignVariable_38save_1/RestoreV2:32*
T0*
_class
loc:@Variable_38*
validate_shape(*
use_locking(

save_1/Assign_33AssignVariable_39save_1/RestoreV2:33*
_class
loc:@Variable_39*
validate_shape(*
use_locking(*
T0

save_1/Assign_34Assign
Variable_4save_1/RestoreV2:34*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(

save_1/Assign_35AssignVariable_40save_1/RestoreV2:35*
_class
loc:@Variable_40*
validate_shape(*
use_locking(*
T0

save_1/Assign_36AssignVariable_41save_1/RestoreV2:36*
_class
loc:@Variable_41*
validate_shape(*
use_locking(*
T0

save_1/Assign_37AssignVariable_42save_1/RestoreV2:37*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_42

save_1/Assign_38AssignVariable_43save_1/RestoreV2:38*
T0*
_class
loc:@Variable_43*
validate_shape(*
use_locking(

save_1/Assign_39AssignVariable_44save_1/RestoreV2:39*
use_locking(*
T0*
_class
loc:@Variable_44*
validate_shape(

save_1/Assign_40AssignVariable_45save_1/RestoreV2:40*
use_locking(*
T0*
_class
loc:@Variable_45*
validate_shape(

save_1/Assign_41AssignVariable_46save_1/RestoreV2:41*
use_locking(*
T0*
_class
loc:@Variable_46*
validate_shape(

save_1/Assign_42AssignVariable_47save_1/RestoreV2:42*
use_locking(*
T0*
_class
loc:@Variable_47*
validate_shape(

save_1/Assign_43AssignVariable_48save_1/RestoreV2:43*
use_locking(*
T0*
_class
loc:@Variable_48*
validate_shape(

save_1/Assign_44AssignVariable_49save_1/RestoreV2:44*
use_locking(*
T0*
_class
loc:@Variable_49*
validate_shape(

save_1/Assign_45Assign
Variable_5save_1/RestoreV2:45*
T0*
_class
loc:@Variable_5*
validate_shape(*
use_locking(

save_1/Assign_46AssignVariable_50save_1/RestoreV2:46*
use_locking(*
T0*
_class
loc:@Variable_50*
validate_shape(

save_1/Assign_47AssignVariable_51save_1/RestoreV2:47*
use_locking(*
T0*
_class
loc:@Variable_51*
validate_shape(

save_1/Assign_48AssignVariable_52save_1/RestoreV2:48*
use_locking(*
T0*
_class
loc:@Variable_52*
validate_shape(

save_1/Assign_49AssignVariable_53save_1/RestoreV2:49*
use_locking(*
T0*
_class
loc:@Variable_53*
validate_shape(

save_1/Assign_50AssignVariable_54save_1/RestoreV2:50*
_class
loc:@Variable_54*
validate_shape(*
use_locking(*
T0

save_1/Assign_51AssignVariable_55save_1/RestoreV2:51*
use_locking(*
T0*
_class
loc:@Variable_55*
validate_shape(

save_1/Assign_52AssignVariable_56save_1/RestoreV2:52*
use_locking(*
T0*
_class
loc:@Variable_56*
validate_shape(

save_1/Assign_53AssignVariable_57save_1/RestoreV2:53*
_class
loc:@Variable_57*
validate_shape(*
use_locking(*
T0

save_1/Assign_54AssignVariable_58save_1/RestoreV2:54*
_class
loc:@Variable_58*
validate_shape(*
use_locking(*
T0

save_1/Assign_55AssignVariable_59save_1/RestoreV2:55*
use_locking(*
T0*
_class
loc:@Variable_59*
validate_shape(

save_1/Assign_56Assign
Variable_6save_1/RestoreV2:56*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(

save_1/Assign_57AssignVariable_60save_1/RestoreV2:57*
use_locking(*
T0*
_class
loc:@Variable_60*
validate_shape(

save_1/Assign_58AssignVariable_61save_1/RestoreV2:58*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_61

save_1/Assign_59AssignVariable_62save_1/RestoreV2:59*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_62

save_1/Assign_60AssignVariable_63save_1/RestoreV2:60*
T0*
_class
loc:@Variable_63*
validate_shape(*
use_locking(

save_1/Assign_61AssignVariable_64save_1/RestoreV2:61*
use_locking(*
T0*
_class
loc:@Variable_64*
validate_shape(

save_1/Assign_62AssignVariable_65save_1/RestoreV2:62*
T0*
_class
loc:@Variable_65*
validate_shape(*
use_locking(

save_1/Assign_63AssignVariable_66save_1/RestoreV2:63*
use_locking(*
T0*
_class
loc:@Variable_66*
validate_shape(

save_1/Assign_64AssignVariable_67save_1/RestoreV2:64*
T0*
_class
loc:@Variable_67*
validate_shape(*
use_locking(

save_1/Assign_65AssignVariable_68save_1/RestoreV2:65*
T0*
_class
loc:@Variable_68*
validate_shape(*
use_locking(

save_1/Assign_66AssignVariable_69save_1/RestoreV2:66*
use_locking(*
T0*
_class
loc:@Variable_69*
validate_shape(

save_1/Assign_67Assign
Variable_7save_1/RestoreV2:67*
_class
loc:@Variable_7*
validate_shape(*
use_locking(*
T0

save_1/Assign_68AssignVariable_70save_1/RestoreV2:68*
use_locking(*
T0*
_class
loc:@Variable_70*
validate_shape(

save_1/Assign_69AssignVariable_71save_1/RestoreV2:69*
use_locking(*
T0*
_class
loc:@Variable_71*
validate_shape(

save_1/Assign_70Assign
Variable_8save_1/RestoreV2:70*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(

save_1/Assign_71Assign
Variable_9save_1/RestoreV2:71*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(

save_1/Assign_72Assignbeta_1save_1/RestoreV2:72*
_class
loc:@beta_1*
validate_shape(*
use_locking(*
T0

save_1/Assign_73Assignbeta_1_1save_1/RestoreV2:73*
use_locking(*
T0*
_class
loc:@beta_1_1*
validate_shape(

save_1/Assign_74Assignbeta_2save_1/RestoreV2:74*
_class
loc:@beta_2*
validate_shape(*
use_locking(*
T0

save_1/Assign_75Assignbeta_2_1save_1/RestoreV2:75*
validate_shape(*
use_locking(*
T0*
_class
loc:@beta_2_1

save_1/Assign_76Assigndecaysave_1/RestoreV2:76*
use_locking(*
T0*
_class

loc:@decay*
validate_shape(

save_1/Assign_77Assigndecay_1save_1/RestoreV2:77*
use_locking(*
T0*
_class
loc:@decay_1*
validate_shape(

save_1/Assign_78Assigndense_1/biassave_1/RestoreV2:78*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(

save_1/Assign_79Assigndense_1/kernelsave_1/RestoreV2:79*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(

save_1/Assign_80Assigndense_2/biassave_1/RestoreV2:80*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(

save_1/Assign_81Assigndense_2/kernelsave_1/RestoreV2:81*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(

save_1/Assign_82Assigndense_3/biassave_1/RestoreV2:82*
validate_shape(*
use_locking(*
T0*
_class
loc:@dense_3/bias

save_1/Assign_83Assigndense_3/kernelsave_1/RestoreV2:83*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(

save_1/Assign_84Assigndense_4/biassave_1/RestoreV2:84*
validate_shape(*
use_locking(*
T0*
_class
loc:@dense_4/bias

save_1/Assign_85Assigndense_4/kernelsave_1/RestoreV2:85*!
_class
loc:@dense_4/kernel*
validate_shape(*
use_locking(*
T0

save_1/Assign_86Assign
iterationssave_1/RestoreV2:86*
use_locking(*
T0*
_class
loc:@iterations*
validate_shape(

save_1/Assign_87Assigniterations_1save_1/RestoreV2:87*
validate_shape(*
use_locking(*
T0*
_class
loc:@iterations_1
|
save_1/Assign_88Assignlrsave_1/RestoreV2:88*
T0*
_class
	loc:@lr*
validate_shape(*
use_locking(

save_1/Assign_89Assignlr_1save_1/RestoreV2:89*
validate_shape(*
use_locking(*
T0*
_class
	loc:@lr_1

save_1/Assign_90Assignres1a_branch2a/biassave_1/RestoreV2:90*
use_locking(*
T0*&
_class
loc:@res1a_branch2a/bias*
validate_shape(
¢
save_1/Assign_91Assignres1a_branch2a/kernelsave_1/RestoreV2:91*(
_class
loc:@res1a_branch2a/kernel*
validate_shape(*
use_locking(*
T0
¢
save_1/Assign_92Assignres1a_branch2a_1/biassave_1/RestoreV2:92*(
_class
loc:@res1a_branch2a_1/bias*
validate_shape(*
use_locking(*
T0
¦
save_1/Assign_93Assignres1a_branch2a_1/kernelsave_1/RestoreV2:93*
T0**
_class 
loc:@res1a_branch2a_1/kernel*
validate_shape(*
use_locking(

save_1/Assign_94Assignres1a_branch2b/biassave_1/RestoreV2:94*
validate_shape(*
use_locking(*
T0*&
_class
loc:@res1a_branch2b/bias
¢
save_1/Assign_95Assignres1a_branch2b/kernelsave_1/RestoreV2:95*(
_class
loc:@res1a_branch2b/kernel*
validate_shape(*
use_locking(*
T0
¢
save_1/Assign_96Assignres1a_branch2b_1/biassave_1/RestoreV2:96*
use_locking(*
T0*(
_class
loc:@res1a_branch2b_1/bias*
validate_shape(
¦
save_1/Assign_97Assignres1a_branch2b_1/kernelsave_1/RestoreV2:97*
use_locking(*
T0**
_class 
loc:@res1a_branch2b_1/kernel*
validate_shape(

save_1/Assign_98Assignres1b_branch2a/biassave_1/RestoreV2:98*
T0*&
_class
loc:@res1b_branch2a/bias*
validate_shape(*
use_locking(
¢
save_1/Assign_99Assignres1b_branch2a/kernelsave_1/RestoreV2:99*
T0*(
_class
loc:@res1b_branch2a/kernel*
validate_shape(*
use_locking(
¤
save_1/Assign_100Assignres1b_branch2a_1/biassave_1/RestoreV2:100*
use_locking(*
T0*(
_class
loc:@res1b_branch2a_1/bias*
validate_shape(
Ø
save_1/Assign_101Assignres1b_branch2a_1/kernelsave_1/RestoreV2:101*
use_locking(*
T0**
_class 
loc:@res1b_branch2a_1/kernel*
validate_shape(
 
save_1/Assign_102Assignres1b_branch2b/biassave_1/RestoreV2:102*
use_locking(*
T0*&
_class
loc:@res1b_branch2b/bias*
validate_shape(
¤
save_1/Assign_103Assignres1b_branch2b/kernelsave_1/RestoreV2:103*
use_locking(*
T0*(
_class
loc:@res1b_branch2b/kernel*
validate_shape(
¤
save_1/Assign_104Assignres1b_branch2b_1/biassave_1/RestoreV2:104*
use_locking(*
T0*(
_class
loc:@res1b_branch2b_1/bias*
validate_shape(
Ø
save_1/Assign_105Assignres1b_branch2b_1/kernelsave_1/RestoreV2:105*
T0**
_class 
loc:@res1b_branch2b_1/kernel*
validate_shape(*
use_locking(
 
save_1/Assign_106Assignres1c_branch2a/biassave_1/RestoreV2:106*
use_locking(*
T0*&
_class
loc:@res1c_branch2a/bias*
validate_shape(
¤
save_1/Assign_107Assignres1c_branch2a/kernelsave_1/RestoreV2:107*
use_locking(*
T0*(
_class
loc:@res1c_branch2a/kernel*
validate_shape(
 
save_1/Assign_108Assignres1c_branch2b/biassave_1/RestoreV2:108*
use_locking(*
T0*&
_class
loc:@res1c_branch2b/bias*
validate_shape(
¤
save_1/Assign_109Assignres1c_branch2b/kernelsave_1/RestoreV2:109*
T0*(
_class
loc:@res1c_branch2b/kernel*
validate_shape(*
use_locking(
 
save_1/Assign_110Assignres1d_branch2a/biassave_1/RestoreV2:110*
use_locking(*
T0*&
_class
loc:@res1d_branch2a/bias*
validate_shape(
¤
save_1/Assign_111Assignres1d_branch2a/kernelsave_1/RestoreV2:111*
use_locking(*
T0*(
_class
loc:@res1d_branch2a/kernel*
validate_shape(
 
save_1/Assign_112Assignres1d_branch2b/biassave_1/RestoreV2:112*
use_locking(*
T0*&
_class
loc:@res1d_branch2b/bias*
validate_shape(
¤
save_1/Assign_113Assignres1d_branch2b/kernelsave_1/RestoreV2:113*
validate_shape(*
use_locking(*
T0*(
_class
loc:@res1d_branch2b/kernel
 
save_1/Assign_114Assignres1e_branch2a/biassave_1/RestoreV2:114*
use_locking(*
T0*&
_class
loc:@res1e_branch2a/bias*
validate_shape(
¤
save_1/Assign_115Assignres1e_branch2a/kernelsave_1/RestoreV2:115*
validate_shape(*
use_locking(*
T0*(
_class
loc:@res1e_branch2a/kernel
 
save_1/Assign_116Assignres1e_branch2b/biassave_1/RestoreV2:116*
T0*&
_class
loc:@res1e_branch2b/bias*
validate_shape(*
use_locking(
¤
save_1/Assign_117Assignres1e_branch2b/kernelsave_1/RestoreV2:117*
validate_shape(*
use_locking(*
T0*(
_class
loc:@res1e_branch2b/kernel
ā
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_100^save_1/Assign_101^save_1/Assign_102^save_1/Assign_103^save_1/Assign_104^save_1/Assign_105^save_1/Assign_106^save_1/Assign_107^save_1/Assign_108^save_1/Assign_109^save_1/Assign_11^save_1/Assign_110^save_1/Assign_111^save_1/Assign_112^save_1/Assign_113^save_1/Assign_114^save_1/Assign_115^save_1/Assign_116^save_1/Assign_117^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_7^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_76^save_1/Assign_77^save_1/Assign_78^save_1/Assign_79^save_1/Assign_8^save_1/Assign_80^save_1/Assign_81^save_1/Assign_82^save_1/Assign_83^save_1/Assign_84^save_1/Assign_85^save_1/Assign_86^save_1/Assign_87^save_1/Assign_88^save_1/Assign_89^save_1/Assign_9^save_1/Assign_90^save_1/Assign_91^save_1/Assign_92^save_1/Assign_93^save_1/Assign_94^save_1/Assign_95^save_1/Assign_96^save_1/Assign_97^save_1/Assign_98^save_1/Assign_99"
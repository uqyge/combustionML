Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

dimensionedScalar nu
(
    "nu",
    dimViscosity,
    transportProperties.lookup("nu")
);

Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


#include "createPhi.H"


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, mesh.solutionDict().subDict("PISO"), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());


const wchar_t* resnet20Model = L"resnet20.dnn";

printf("\n##### Run CNTKLibraryCPPEvalCPUOnlyExamples on CPU. #####\n");

// Evaluate a single image with reset20_cifar model.
EvaluationSingleSampleUsingDense(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());

// Evaluate batch of images with reset20_cifar model.
EvaluationBatchUsingDense(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());

// Evaluate users requests in parallel with reset20_cifar model.
ParallelEvaluationExample(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());

printf("Evaluation complete.\n");

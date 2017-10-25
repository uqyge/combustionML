CNTK_COMPONENT_VERSION := 2.2
CNTKLIBRARY:=Cntk.Core-$(CNTK_COMPONENT_VERSION)

LIBPATH:= /cntk/cntk/dependencies/lib
LIBPATH+= /cntk/cntk/lib

RPATH=-Wl,-rpath,

CXX=mpic++
# SRCDIR=/home/openfoam/Evaluation/CNTKLibraryCPPEvalCPUOnlyExamples/
SRCDIR=./
CXXFLAGS += -g -O4
LDFLAGS += -rdynamic

EvalExample: CNTKLibraryCPPEvalExamples.o CNTKLibraryCPPEvalCPUOnlyExamples.o
	@echo "---------------------"
	@mkdir -p $(dir $@)
	@echo building $(CNTKLIBRARY_CPP_EVAL_EXAMPLES) for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH)) $(patsubst %,$(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -l$(CNTKLIBRARY)
	cp CNTKLibraryCPPEvalCPUOnlyExamples.o libCNTKLibraryCPPEvalCPUOnlyExamples.so 

CNTKLibraryCPPEvalExamples.o: $(SRCDIR)/CNTKLibraryCPPEvalExamples.cpp
	$(CXX) -msse4.1 -mssse3 -std=c++0x -fopenmp -fpermissive -fPIC -Werror -fcheck-new -Wno-error=literal-suffix -g -O4  -DCNTK_COMPONENT_VERSION="2.2" -c $^ -o $(SRCDIR)/$@ -I/cntk/Include

CNTKLibraryCPPEvalCPUOnlyExamples.o: $(SRCDIR)/CNTKLibraryCPPEvalCPUOnlyExamples.cpp
	$(CXX) -msse4.1 -mssse3 -std=c++0x -fopenmp -fpermissive -fPIC -Werror -fcheck-new -Wno-error=literal-suffix -g -O4  -DCNTK_COMPONENT_VERSION="2.2" -c $^ -o $(SRCDIR)/$@ -I/cntk/Include

clean:
	rm *.o

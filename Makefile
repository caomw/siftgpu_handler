#############################################################################
# Makefile for building: libsift.1.0.0.dylib
# Generated by qmake (2.01a) (Qt 4.8.6) on: Thu Jan 29 13:08:54 2015
# Project:  sift.pro
# Template: lib
# Command: /usr/bin/qmake -spec /usr/local/Qt4.8/mkspecs/macx-g++ CONFIG+=release CONFIG+=x86_64 INCLUDEPATH+=/Users/STW/Documents/Projects/tianweibundle/thirdpartylibs/install/include LIBS+=-L/Users/STW/Documents/Projects/tianweibundle/thirdpartylibs/install/lib -o Makefile sift.pro
#############################################################################

####### Compiler, tools and options

CC            = gcc
CXX           = g++
DEFINES       = -DZETA_SIFTGPU_ENABLE -DZETA_OPENGL_ENABLE -DQT_NO_DEBUG
CFLAGS        = -pipe -O3 -O2 -arch x86_64 -Xarch_x86_64 -mmacosx-version-min=10.5 -Wall -W -fPIC $(DEFINES)
CXXFLAGS      = -pipe -c -fmessage-length=0 -fomit-frame-pointer -O3 -O2 -arch x86_64 -Xarch_x86_64 -mmacosx-version-min=10.5 -Wall -W -fPIC $(DEFINES)
INCPATH       = -I/usr/local/Qt4.8/mkspecs/macx-g++ -I. -I/usr/include -I../thirdpartylibs/install/include -I../tianweibundle/libs -I.. -I/usr/local/include -I/usr/include -I/opt/local/include -I../tianweibundle -Iout -I.
LINK          = g++
LFLAGS        = -headerpad_max_install_names -framework OpenGL -framework GLUT -arch x86_64 -single_module -dynamiclib -compatibility_version	1.0 -current_version	1.0.0 -install_name	libsift.1.dylib -Xarch_x86_64 -mmacosx-version-min=10.5
LIBS          = $(SUBLIBS)  -L/Library/Frameworks -L/Users/STW/Documents/Projects/tianweibundle/thirdpartylibs/install/lib -L/Users/STW/Documents/Projects/tianweibundle/tianweibundle/libs -L/opt/local/lib -L/usr/lib -L/usr/local/lib -L/Users/STW/Documents/Projects/tianweibundle/tianweibundle/libs/siftgpu -lsiftgpu -L../../Base -lBase 
AR            = ar cq
RANLIB        = ranlib -s
QMAKE         = /usr/bin/qmake
TAR           = tar -cf
COMPRESS      = gzip -9f
COPY          = cp -f
SED           = sed
COPY_FILE     = cp -f
COPY_DIR      = cp -f -R
STRIP         = 
INSTALL_FILE  = $(COPY_FILE)
INSTALL_DIR   = $(COPY_DIR)
INSTALL_PROGRAM = $(COPY_FILE)
DEL_FILE      = rm -f
SYMLINK       = ln -f -s
DEL_DIR       = rmdir
MOVE          = mv -f
CHK_DIR_EXISTS= test -d
MKDIR         = mkdir -p
export MACOSX_DEPLOYMENT_TARGET = 10.4

####### Output directory

OBJECTS_DIR   = out/

####### Files

SOURCES       = commandHandler.cpp \
		featureModule.cpp \
		matchModule.cpp 
OBJECTS       = out/commandHandler.o \
		out/featureModule.o \
		out/matchModule.o
DIST          = /usr/local/Qt4.8/mkspecs/common/unix.conf \
		/usr/local/Qt4.8/mkspecs/common/mac.conf \
		/usr/local/Qt4.8/mkspecs/common/gcc-base.conf \
		/usr/local/Qt4.8/mkspecs/common/gcc-base-macx.conf \
		/usr/local/Qt4.8/mkspecs/common/g++-base.conf \
		/usr/local/Qt4.8/mkspecs/common/g++-macx.conf \
		/usr/local/Qt4.8/mkspecs/features/exclusive_builds.prf \
		/usr/local/Qt4.8/mkspecs/features/default_pre.prf \
		/usr/local/Qt4.8/mkspecs/features/mac/default_pre.prf \
		../tianweibundle/.qmake.cache \
		/usr/local/Qt4.8/mkspecs/qconfig.pri \
		/usr/local/Qt4.8/mkspecs/modules/qt_webkit_version.pri \
		/usr/local/Qt4.8/mkspecs/features/qt_functions.prf \
		/usr/local/Qt4.8/mkspecs/features/qt_config.prf \
		../baselib/BaseQTFlags.pri \
		../baselib/libraryDetector.pri \
		../baselib/QTFlagsSiftgpu.pri \
		../baselib/QTFlagsANNmt.pri \
		../baselib/QTFlagsOpenGL.pri \
		/usr/local/Qt4.8/mkspecs/features/release.prf \
		/usr/local/Qt4.8/mkspecs/features/default_post.prf \
		/usr/local/Qt4.8/mkspecs/features/mac/default_post.prf \
		/usr/local/Qt4.8/mkspecs/features/mac/x86_64.prf \
		/usr/local/Qt4.8/mkspecs/features/mac/objective_c.prf \
		/usr/local/Qt4.8/mkspecs/features/shared.prf \
		/usr/local/Qt4.8/mkspecs/features/dll.prf \
		/usr/local/Qt4.8/mkspecs/features/warn_on.prf \
		/usr/local/Qt4.8/mkspecs/features/qt.prf \
		/usr/local/Qt4.8/mkspecs/features/unix/thread.prf \
		/usr/local/Qt4.8/mkspecs/features/moc.prf \
		/usr/local/Qt4.8/mkspecs/features/mac/rez.prf \
		/usr/local/Qt4.8/mkspecs/features/mac/sdk.prf \
		/usr/local/Qt4.8/mkspecs/features/resources.prf \
		/usr/local/Qt4.8/mkspecs/features/uic.prf \
		/usr/local/Qt4.8/mkspecs/features/yacc.prf \
		/usr/local/Qt4.8/mkspecs/features/lex.prf \
		sift.pro
QMAKE_TARGET  = sift
DESTDIR       = 
TARGET        = libsift.1.0.0.dylib
TARGETA       = libsift.a
TARGETD       = libsift.1.0.0.dylib
TARGET0       = libsift.dylib
TARGET1       = libsift.1.dylib
TARGET2       = libsift.1.0.dylib

####### Custom Compiler Variables
QMAKE_COMP_QMAKE_OBJECTIVE_CFLAGS = -pipe \
		-O2 \
		-arch \
		x86_64 \
		-Xarch_x86_64 \
		-mmacosx-version-min=10.5 \
		-Wall \
		-W


first: all
####### Implicit rules

.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules

all: Makefile  $(TARGET)

$(TARGET):  $(OBJECTS) $(SUBLIBS) $(OBJCOMP)  
	-$(DEL_FILE) $(TARGET) $(TARGET0) $(TARGET1) $(TARGET2)
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(LIBS) $(OBJCOMP)
	-ln -s $(TARGET) $(TARGET0)
	-ln -s $(TARGET) $(TARGET1)
	-ln -s $(TARGET) $(TARGET2)



staticlib: $(TARGETA)

$(TARGETA):  $(OBJECTS) $(OBJCOMP) 
	-$(DEL_FILE) $(TARGETA) 
	$(AR) $(TARGETA) $(OBJECTS)
	$(RANLIB) $(TARGETA)

Makefile: sift.pro ../tianweibundle/.qmake.cache /usr/local/Qt4.8/mkspecs/macx-g++/qmake.conf /usr/local/Qt4.8/mkspecs/common/unix.conf \
		/usr/local/Qt4.8/mkspecs/common/mac.conf \
		/usr/local/Qt4.8/mkspecs/common/gcc-base.conf \
		/usr/local/Qt4.8/mkspecs/common/gcc-base-macx.conf \
		/usr/local/Qt4.8/mkspecs/common/g++-base.conf \
		/usr/local/Qt4.8/mkspecs/common/g++-macx.conf \
		/usr/local/Qt4.8/mkspecs/features/exclusive_builds.prf \
		/usr/local/Qt4.8/mkspecs/features/default_pre.prf \
		/usr/local/Qt4.8/mkspecs/features/mac/default_pre.prf \
		../tianweibundle/.qmake.cache \
		/usr/local/Qt4.8/mkspecs/qconfig.pri \
		/usr/local/Qt4.8/mkspecs/modules/qt_webkit_version.pri \
		/usr/local/Qt4.8/mkspecs/features/qt_functions.prf \
		/usr/local/Qt4.8/mkspecs/features/qt_config.prf \
		../baselib/BaseQTFlags.pri \
		../baselib/libraryDetector.pri \
		../baselib/QTFlagsSiftgpu.pri \
		../baselib/QTFlagsANNmt.pri \
		../baselib/QTFlagsOpenGL.pri \
		/usr/local/Qt4.8/mkspecs/features/release.prf \
		/usr/local/Qt4.8/mkspecs/features/default_post.prf \
		/usr/local/Qt4.8/mkspecs/features/mac/default_post.prf \
		/usr/local/Qt4.8/mkspecs/features/mac/x86_64.prf \
		/usr/local/Qt4.8/mkspecs/features/mac/objective_c.prf \
		/usr/local/Qt4.8/mkspecs/features/shared.prf \
		/usr/local/Qt4.8/mkspecs/features/dll.prf \
		/usr/local/Qt4.8/mkspecs/features/warn_on.prf \
		/usr/local/Qt4.8/mkspecs/features/qt.prf \
		/usr/local/Qt4.8/mkspecs/features/unix/thread.prf \
		/usr/local/Qt4.8/mkspecs/features/moc.prf \
		/usr/local/Qt4.8/mkspecs/features/mac/rez.prf \
		/usr/local/Qt4.8/mkspecs/features/mac/sdk.prf \
		/usr/local/Qt4.8/mkspecs/features/resources.prf \
		/usr/local/Qt4.8/mkspecs/features/uic.prf \
		/usr/local/Qt4.8/mkspecs/features/yacc.prf \
		/usr/local/Qt4.8/mkspecs/features/lex.prf
	$(QMAKE) -spec /usr/local/Qt4.8/mkspecs/macx-g++ CONFIG+=release CONFIG+=x86_64 INCLUDEPATH+=/Users/STW/Documents/Projects/tianweibundle/thirdpartylibs/install/include LIBS+=-L/Users/STW/Documents/Projects/tianweibundle/thirdpartylibs/install/lib -o Makefile sift.pro
/usr/local/Qt4.8/mkspecs/common/unix.conf:
/usr/local/Qt4.8/mkspecs/common/mac.conf:
/usr/local/Qt4.8/mkspecs/common/gcc-base.conf:
/usr/local/Qt4.8/mkspecs/common/gcc-base-macx.conf:
/usr/local/Qt4.8/mkspecs/common/g++-base.conf:
/usr/local/Qt4.8/mkspecs/common/g++-macx.conf:
/usr/local/Qt4.8/mkspecs/features/exclusive_builds.prf:
/usr/local/Qt4.8/mkspecs/features/default_pre.prf:
/usr/local/Qt4.8/mkspecs/features/mac/default_pre.prf:
../tianweibundle/.qmake.cache:
/usr/local/Qt4.8/mkspecs/qconfig.pri:
/usr/local/Qt4.8/mkspecs/modules/qt_webkit_version.pri:
/usr/local/Qt4.8/mkspecs/features/qt_functions.prf:
/usr/local/Qt4.8/mkspecs/features/qt_config.prf:
../baselib/BaseQTFlags.pri:
../baselib/libraryDetector.pri:
../baselib/QTFlagsSiftgpu.pri:
../baselib/QTFlagsANNmt.pri:
../baselib/QTFlagsOpenGL.pri:
/usr/local/Qt4.8/mkspecs/features/release.prf:
/usr/local/Qt4.8/mkspecs/features/default_post.prf:
/usr/local/Qt4.8/mkspecs/features/mac/default_post.prf:
/usr/local/Qt4.8/mkspecs/features/mac/x86_64.prf:
/usr/local/Qt4.8/mkspecs/features/mac/objective_c.prf:
/usr/local/Qt4.8/mkspecs/features/shared.prf:
/usr/local/Qt4.8/mkspecs/features/dll.prf:
/usr/local/Qt4.8/mkspecs/features/warn_on.prf:
/usr/local/Qt4.8/mkspecs/features/qt.prf:
/usr/local/Qt4.8/mkspecs/features/unix/thread.prf:
/usr/local/Qt4.8/mkspecs/features/moc.prf:
/usr/local/Qt4.8/mkspecs/features/mac/rez.prf:
/usr/local/Qt4.8/mkspecs/features/mac/sdk.prf:
/usr/local/Qt4.8/mkspecs/features/resources.prf:
/usr/local/Qt4.8/mkspecs/features/uic.prf:
/usr/local/Qt4.8/mkspecs/features/yacc.prf:
/usr/local/Qt4.8/mkspecs/features/lex.prf:
qmake:  FORCE
	@$(QMAKE) -spec /usr/local/Qt4.8/mkspecs/macx-g++ CONFIG+=release CONFIG+=x86_64 INCLUDEPATH+=/Users/STW/Documents/Projects/tianweibundle/thirdpartylibs/install/include LIBS+=-L/Users/STW/Documents/Projects/tianweibundle/thirdpartylibs/install/lib -o Makefile sift.pro

dist: 
	@$(CHK_DIR_EXISTS) out/sift1.0.0 || $(MKDIR) out/sift1.0.0 
	$(COPY_FILE) --parents $(SOURCES) $(DIST) out/sift1.0.0/ && $(COPY_FILE) --parents commandHandler.h featureModule.h matchModule.h out/sift1.0.0/ && $(COPY_FILE) --parents commandHandler.cpp featureModule.cpp matchModule.cpp out/sift1.0.0/ && (cd `dirname out/sift1.0.0` && $(TAR) sift1.0.0.tar sift1.0.0 && $(COMPRESS) sift1.0.0.tar) && $(MOVE) `dirname out/sift1.0.0`/sift1.0.0.tar.gz . && $(DEL_FILE) -r out/sift1.0.0


clean:compiler_clean 
	-$(DEL_FILE) $(OBJECTS)
	-$(DEL_FILE) *~ core *.core


####### Sub-libraries

distclean: clean
	-$(DEL_FILE) $(TARGET) 
	-$(DEL_FILE) $(TARGET0) $(TARGET1) $(TARGET2) $(TARGETA)
	-$(DEL_FILE) Makefile


check: first

mocclean: compiler_moc_header_clean compiler_moc_source_clean

mocables: compiler_moc_header_make_all compiler_moc_source_make_all

compiler_objective_c_make_all:
compiler_objective_c_clean:
compiler_moc_header_make_all:
compiler_moc_header_clean:
compiler_rcc_make_all:
compiler_rcc_clean:
compiler_image_collection_make_all: ../tianweibundle/Handlers/sift/out/qmake_image_collection.cpp
compiler_image_collection_clean:
	-$(DEL_FILE) ../tianweibundle/Handlers/sift/out/qmake_image_collection.cpp
compiler_moc_source_make_all:
compiler_moc_source_clean:
compiler_rez_source_make_all:
compiler_rez_source_clean:
compiler_uic_make_all:
compiler_uic_clean:
compiler_yacc_decl_make_all:
compiler_yacc_decl_clean:
compiler_yacc_impl_make_all:
compiler_yacc_impl_clean:
compiler_lex_make_all:
compiler_lex_clean:
compiler_clean: 

####### Compile

out/commandHandler.o: commandHandler.cpp commandHandler.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o out/commandHandler.o commandHandler.cpp

out/featureModule.o: featureModule.cpp featureModule.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o out/featureModule.o featureModule.cpp

out/matchModule.o: matchModule.cpp featureModule.h \
		matchModule.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o out/matchModule.o matchModule.cpp

####### Install

install:   FORCE

uninstall:   FORCE

FORCE:

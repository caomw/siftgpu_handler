#-------------------------------------------------
#
# Project created by QtCreator 2012-12-20T20:21:17
#
#-------------------------------------------------

QT       -= core gui

# TARGET = stereotuner # by default, it is the project name

TEMPLATE = lib

SOURCES += $$files(*.cpp)

HEADERS += $$files(*.h)

!include($${top_srcdir}/Base/BaseQTFlags.pri) {
    error("Cannot include BaseQTFlags.pri")
}

SIFTGPUENABLE = 1
!include($${top_srcdir}/Base/QTFlagsSiftgpu.pri) {
    error("Cannot include QTFlagsSiftgpu.pri")
}

!include($${top_srcdir}/Base/QTFlagsANNmt.pri) {
    error("Cannot include QTFlagsANNmt.pri")
}

OPENGLENABLE = 1
!include($${top_srcdir}/Base/QTFlagsOpenGL.pri) {
    error("Cannot include QTFlagsOpenGL.pri")
}

INCLUDEPATH += $${top_srcdir}

macx | linux-g++* | win32 {
    LIBS += -L../../Base -lBase$$DEBUGSUFFIX
}

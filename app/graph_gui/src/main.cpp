//
// Created by mingkaichen on 11/16/17.
//

#include <sstream>
#include <memory>

#include <QtWidgets/QApplication>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QMenuBar>

#include <nodes/Node>
#include <nodes/FlowScene>
#include <nodes/FlowView>
#include <nodes/ConnectionStyle>

#include "qthread_wrap.hpp"
#include "gui_consumer.hpp"
#include "node_model.hpp"

static std::shared_ptr<QtNodes::DataModelRegistry> registerDataModels (void)
{
	auto ret = std::make_shared<QtNodes::DataModelRegistry>();
	ret->registerModel<tenncor_graph::node_model>("Node");

	return ret;
}

static void setStyle (void)
{
	QtNodes::ConnectionStyle::setConnectionStyle(
	R"({
		"ConnectionStyle":
		{
			"ConstructionColor": "gray",
			"NormalColor": "black",
			"SelectedColor": "gray",
			"SelectedHaloColor": "deepskyblue",
			"HoveredColor": "deepskyblue",

			"LineWidth": 3.0,
			"ConstructionLineWidth": 2.0,
			"PointDiameter": 10.0,

			"UseDataDefinedColors": true
		}
	})");
}

int main(int argc, char** argv)
{
	std::stringstream addr;
	addr << "localhost:" << 50981;
	tenncor_graph::gui_consumer consumer(grpc::CreateChannel(addr.str(), grpc::InsecureChannelCredentials()));

	QApplication app(argc, argv);
	setStyle();
	QWidget mainWidget;
	auto menuBar = new QMenuBar();
	auto saveAction = menuBar->addAction("Save..");
	auto loadAction = menuBar->addAction("Load..");

	QVBoxLayout* l = new QVBoxLayout(&mainWidget);

	l->addWidget(menuBar);
	auto scene = new QtNodes::FlowScene(registerDataModels());
	auto view = new QtNodes::FlowView(scene);
	l->addWidget(view);
	l->setContentsMargins(0, 0, 0, 0);
	l->setSpacing(0);

	QObject::connect(saveAction, &QAction::triggered,
		scene, &QtNodes::FlowScene::save);
	QObject::connect(loadAction, &QAction::triggered,
		scene, &QtNodes::FlowScene::load);

	mainWidget.setWindowTitle("Tenncor Operations Graph");
	mainWidget.resize(800, 600);
	mainWidget.showNormal();

	// do something to add nodes to the scene + view
	tenncor_graph::qthread_wrap nodeHandler(
	[](tenncor_graph::gui_consumer& consumer)
	{
		consumer.SubscribeNode();
	}, std::ref(consumer));

	tenncor_graph::qthread_wrap edgeHandler(
	[](tenncor_graph::gui_consumer& consumer)
	{
		consumer.SubscribeEdge();
	}, std::ref(consumer));

	int exit_status = app.exec();

	nodeHandler.wait();
	edgeHandler.wait();

	return exit_status;
}
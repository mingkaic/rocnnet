//
// Created by mingkaichen on 11/16/17.
//

#pragma once

#include <QtCore/QObject>
#include <QtWidgets/QLabel>

#include <nodes/NodeDataModel>

#ifndef NODE_MODEL_HPP
#define NODE_MODEL_HPP

namespace tenncor_graph
{

class node_model : public QtNodes::NodeDataModel
{
Q_OBJECT
public:
	node_model (void);

	virtual ~node_model (void);

	QString caption(void) const override;

	bool captionVisible (void) const override;

	QString name (void) const override;

	std::unique_ptr<QtNodes::NodeDataModel> clone (void) const override;

	unsigned int nPorts (QtNodes::PortType portType) const override;

	QtNodes::NodeDataType dataType (QtNodes::PortType portType,
		QtNodes::PortIndex portIndex) const override;

	std::shared_ptr<QtNodes::NodeData> outData (QtNodes::PortIndex port) override;

	void setInData (std::shared_ptr<QtNodes::NodeData> data, int idx) override;

	QWidget* embeddedWidget (void) override;

	QtNodes::NodeValidationState validationState (void) const override;

	QString validationMessage (void) const override;

private:
	QtNodes::NodeValidationState modelValidationState = QtNodes::NodeValidationState::Warning;

	QString modelValidationError = QStringLiteral("Missing or incorrect inputs");

	QLabel* _label;
};

}

#endif /* NODE_MODEL_HPP */

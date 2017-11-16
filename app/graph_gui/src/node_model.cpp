//
// Created by mingkaichen on 11/16/17.
//

#include "node_model.hpp"

#ifdef NODE_MODEL_HPP

namespace tenncor_graph
{

class data_impl : public QtNodes::NodeData
{
public:

	data_impl (void) : _number(0.0) {}

	data_impl (double const number) : _number(number) {}

	QtNodes::NodeDataType type (void) const override
	{
		return QtNodes::NodeDataType {"decimal", "Decimal"};
	}

	double number (void) const
	{
		return _number;
	}

	QString numberAsText (void) const
	{
		return QString::number(_number, 'f');
	}

private:
	double _number;
};

node_model::node_model (void) :
	_label(new QLabel())
{
	_label->setMargin(3);
}

node_model::~node_model (void) {}

QString node_model::caption(void) const
{
	return QStringLiteral("Result");
}

bool node_model::captionVisible (void) const
{
	return false;
}

QString node_model::name (void) const
{
	return QStringLiteral("Result");
}

std::unique_ptr<QtNodes::NodeDataModel> node_model::clone (void) const
{
	return std::make_unique<node_model>();
}

unsigned int node_model::nPorts (QtNodes::PortType portType) const
{
	unsigned int result = 1;

	switch (portType)
	{
		case QtNodes::PortType::In:
			result = 2;
			break;

		case QtNodes::PortType::Out:
			result = 1;

		default:
			break;
	}

	return result;
}

QtNodes::NodeDataType node_model::dataType (QtNodes::PortType, QtNodes::PortIndex) const
{
	return data_impl().type();
}

std::shared_ptr<QtNodes::NodeData> node_model::outData (QtNodes::PortIndex)
{
	std::shared_ptr<QtNodes::NodeData> ptr;
	return ptr;
}

void node_model::setInData (std::shared_ptr<QtNodes::NodeData> data, int)
{
	auto numberData = std::dynamic_pointer_cast<data_impl>(data);

	if (numberData)
	{
		modelValidationState = QtNodes::NodeValidationState::Valid;
		modelValidationError = QString();
		_label->setText(numberData->numberAsText());
	}
	else
	{
		modelValidationState = QtNodes::NodeValidationState::Warning;
		modelValidationError = QStringLiteral("Missing or incorrect inputs");
		_label->clear();
	}

	_label->adjustSize();
}

QWidget* node_model::embeddedWidget (void)
{
	return _label;
}

QtNodes::NodeValidationState node_model::validationState (void) const
{
	return modelValidationState;
}

QString node_model::validationMessage (void) const
{
	return modelValidationError;
}

}

#endif

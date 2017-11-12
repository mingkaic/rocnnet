//
// Created by Ming Kai Chen on 2017-11-11.
//

#include "graph_gui.hpp"

#ifdef graph_gui_hpp

namespace tenncor_graph
{

class graph_gui::CallHandler
{
public:
	// Take in the "service" instance (in this case representing an asynchronous
	// server) and the completion queue "cq" used for asynchronous communication
	// with the gRPC runtime.
	CallHandler (void) :
		responder_(&ctx_), status_(CREATE) {}

	virtual ~CallHandler (void) {}

	void run (void)
	{
		if (status_ == CREATE)
		{
			// request one of the two grpcs in GraphNotifier Service
			this->setup();

			// proceed to next state
			status_ = PROCESS;
		}
		else if (status_ == PROCESS)
		{
			// spawn a new instance and perform processing
			this->process();

			// Let the gRPC runtime know we've finished, using the
			// memory address of this instance as the uniquely identifying tag for
			// the event.
			status_ = FINISH;
			responder_.Finish(reply_, grpc::Status::OK, this);
		}
		else
		{
			GPR_ASSERT(status_ == FINISH);
			// Once in the FINISH state, deallocate ourselves (CallHandler).
			delete this;
		}
	}

protected:
	virtual void setup (void) = 0;

	virtual void process (void) = 0;

	// Context for the rpc, allowing to tweak aspects of it such as the use
	// of compression, authentication, as well as to send metadata back to the
	// client.
	grpc::ServerContext ctx_;

	// The means to get back to the client.
	grpc::ServerAsyncResponseWriter<visor::Empty> responder_;

private:
	// this object's state
	enum CallStatus
	{
		CREATE,
		PROCESS,
		FINISH
	};

	// What we send back to the client.
	visor::Empty reply_;

	// The current serving state.
	CallStatus status_;
};

class graph_gui::NodeHandler : public graph_gui::CallHandler
{
public:
	NodeHandler (visor::GUINotifier::AsyncService* service, grpc::ServerCompletionQueue* cq) :
		CallHandler(), service_(service), cq_(cq)
	{
		// Invoke the serving logic right away.
		this->run();
	}

protected:
	virtual void setup (void)
	{
		service_->RequestNodeChange(&this->ctx_, &request_,
			&this->responder_, this->cq_, this->cq_, this);
	}

	virtual void process (void)
	{
		new NodeHandler(this->service_, this->cq_);

		std::cout << "NODE UPDATE " << request_.id() << std::endl;
	}

private:
	// The means of communication with the gRPC runtime for an asynchronous
	// server.
	visor::GUINotifier::AsyncService* service_;

	// The producer-consumer queue where for asynchronous server notifications.
	grpc::ServerCompletionQueue* cq_;

	// What we get from the client.
	visor::NodeMessage request_;
};

class graph_gui::EdgeHandler : public graph_gui::CallHandler
{
public:
	EdgeHandler (visor::GUINotifier::AsyncService* service, grpc::ServerCompletionQueue* cq) :
		CallHandler(), service_(service), cq_(cq)
	{
		// Invoke the serving logic right away.
		this->run();
	}

protected:
	virtual void setup (void)
	{
		service_->RequestEdgeChange(&this->ctx_, &request_,
			&this->responder_, this->cq_, this->cq_, this);
	}

	virtual void process (void)
	{
		new EdgeHandler(this->service_, this->cq_);

		std::cout << "EDGE UPDATE " << request_.obsid() << " + " << request_.subid() << std::endl;
	}

private:
	// The means of communication with the gRPC runtime for an asynchronous
	// server.
	visor::GUINotifier::AsyncService* service_;

	// The producer-consumer queue where for asynchronous server notifications.
	grpc::ServerCompletionQueue* cq_;

	// What we get from the client.
	visor::EdgeMessage request_;
};

graph_gui::~graph_gui (void)
{
	server_->Shutdown();
	cq_->Shutdown();
}

void graph_gui::run (std::string host, size_t port)
{
	std::stringstream server_addr;
	server_addr << host << ":" << port;

	grpc::ServerBuilder builder;
	// listen on server_addr without auth
	builder.AddListeningPort(server_addr.str(),
		grpc::InsecureServerCredentials());
	// register GUINotifier service
	builder.RegisterService(&service_);
	// store completion queue
	cq_ = builder.AddCompletionQueue();
	// assemble server
	server_ = builder.BuildAndStart();
	std::cout << "GUI Server listening on " << server_addr.str() << std::endl;

	new NodeHandler(&service_, cq_.get());
	new EdgeHandler(&service_, cq_.get());
	void* tag;
	bool ok;
	while (true)
	{
		GPR_ASSERT(cq_->Next(&tag, &ok));
		GPR_ASSERT(ok);
		static_cast<CallHandler*>(tag)->run();
	}
}

}

int main(int argc, char** argv)
{
	tenncor_graph::graph_gui server;
	server.run("localhost", 10980);

	return 0;
}

#endif
